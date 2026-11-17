# SPDX-License-Identifier: MIT
"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Method, in one paragraph: a launch day on Twitch is not a random walk — it is
a rolling midnight unlock. Hour 1 (00:00-01:00 UTC) only has the regions that
already unlocked, so it sits low; the US unlocks lift the morning, and the
curve climbs to an evening peak before easing off. This program learns that
shape from the case's own comparables — past big launches' hourly curves —
instead of assuming it. For each forecast hour it regresses the log-ratio to
hour 1 on log hour-1 size, weighting past launches by how close their hour-1
size is to this case's (a Gaussian kernel, bandwidth 1.5 in log units), after
dropping the weakest 15% of comparables by peak ratio (flops and thin
categories whose hour-1 print was barely a launch at all). The quantile
spread comes from that hour's regression residuals, capped at 0.6 log units
and scaled by 0.65 — the raw residual spread across all comparables overstates
the uncertainty for a big launch, and an honest but tighter band scores
better on the pinball loss. Everything is exact arithmetic on the case file;
no randomness, no network, no wall clock, no files outside the input, so two
runs on the same case write byte-identical forecasts.

Why log hour-1 size matters: across the comparables the evening peak is
almost independent of the hour-1 print (log peak ~ 11.7 + 0.07 * log p1), so
a small hour-1 print means a much bigger rise ahead, and a huge hour-1
print means the launch is already near its peak. A flat persistence forecast
misses that entirely, which is why it scores ~0.387 on the practice cases
while this one scores ~0.15. One guardrail: the prediction's log hour-1 size
is capped at the 90th percentile of the comparables, because the only two
comparables bigger than that had simultaneous global unlocks and peaked in
hour 1 — a pattern a rolling-midnight console launch will not repeat, so
letting a very large GTA VI hour-1 print extrapolate toward "peak already
passed" would be the wrong shape.

Run as the grader runs it: `python3 -P -s /bot/main.py` (Python 3.13, stdlib
only). It reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input/) and writes
one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output/):
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, hours 2..24.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (none is shipped; the line is
# harmless and keeps the layout the starter documents).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # standard normal quantiles at LEVELS

# Kernel bandwidth (log units) for weighting comparables by hour-1 size.
BANDWIDTH = 1.5
# Centre of the log hour-1 size (log viewers); only shifts the intercept.
LOG_P1_CENTRE = 10.8
# Quantile spread: residual SD per hour, clamped, then scaled.
SPREAD_SCALE = 0.65
SD_FLOOR = 0.35
SD_CAP = 0.6
# Drop the weakest share of comparables by peak ratio before fitting.
DROP_BOTTOM = 0.15
MIN_ROWS_PER_HOUR = 8


def round2(x):
    """Round to 2 decimals, half up (the starter's convention)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_points(comp):
    """{UTC hour 0..23: viewers} on the comparable's UTC release day."""
    try:
        day = parse_dt(comp["day_start"]).date()
    except (KeyError, TypeError, ValueError):
        return {}
    pts = {}
    for p in comp.get("hourly") or []:
        try:
            t = parse_dt(p["from"])
            v = p["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if t.date() == day and isinstance(v, (int, float)) and v and v > 0:
            pts[t.hour] = float(v)
    return pts


def comparable_rows(case):
    """One row per usable comparable: hour-1 print and per-hour log-ratios.

    Skips the target category's own pre-release history (not a launch), and
    any comparable without a usable hour-1 print or with too few release-day
    hours. Log-ratio for forecast hour h uses the point at UTC hour h-1.
    """
    rows = []
    for comp in case.get("comparables") or []:
        name = (comp.get("name") or "").lower()
        unlock = (comp.get("unlock") or "").lower()
        if "before release" in name or "not a launch" in unlock:
            continue
        pts = release_day_points(comp)
        if 0 not in pts or pts[0] <= 0:
            continue
        p1 = pts[0]
        lr = {}
        for h in range(2, 25):
            if (h - 1) in pts:
                lr[h] = math.log(pts[h - 1] / p1)
        if len(lr) >= 20:
            peak = max((math.exp(v) for v in lr.values()), default=0.0)
            rows.append({"p1": p1, "lr": lr, "peak_ratio": peak})
    return rows


def solve2(a11, a12, a22, b1, b2):
    """Solve a 2x2 symmetric system; None when singular."""
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return None
    return ((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det)


def fit_hour(rows, p1, h, cap=None):
    """Kernel-weighted OLS of this hour's log-ratio on centred log p1.

    Returns (median log-ratio, spread in log units) or None. ``cap`` is an
    upper bound on the log hour-1 size used for the prediction: past the
    largest launches the comparables stop being informative about the shape
    (the two biggest had simultaneous global unlocks and peaked in hour 1,
    while a rolling-midnight console launch keeps rising into the evening),
    so an hour-1 print above the cap is predicted as if it sat at the cap.
    """
    x0 = math.log(p1) - LOG_P1_CENTRE
    if cap is not None:
        x0 = min(x0, cap - LOG_P1_CENTRE)
    s_w = s_wx = s_wxx = s_wy = s_wxy = 0.0
    pts = []
    x0_log = x0 + LOG_P1_CENTRE
    for r in rows:
        if h not in r["lr"]:
            continue
        x = math.log(r["p1"]) - LOG_P1_CENTRE
        y = r["lr"][h]
        w = math.exp(-0.5 * ((math.log(r["p1"]) - x0_log) / BANDWIDTH) ** 2)
        s_w += w
        s_wx += w * x
        s_wxx += w * x * x
        s_wy += w * y
        s_wxy += w * x * y
        pts.append((x, y))
    if len(pts) < MIN_ROWS_PER_HOUR or s_w <= 0:
        return None
    beta = solve2(s_w, s_wx, s_wxx, s_wy, s_wxy)
    if beta is None:
        return None
    a, b = beta
    mu = a + b * x0
    ss = sum((y - (a + b * x)) ** 2 for x, y in pts)
    sd = math.sqrt(ss / max(len(pts) - 2, 1))
    sd = min(max(sd, SD_FLOOR), SD_CAP)
    return mu, sd


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None or not first.get("viewers"):
        # Defensive fallback: last pre-checkpoint hourly point, else 1.
        hist = case.get("history_hourly") or []
        p1 = float(hist[-1]["viewers"]) if hist and hist[-1].get("viewers") else 1.0
    else:
        p1 = float(first["viewers"])
    p1 = max(1.0, p1)

    rows = comparable_rows(case)
    if rows:
        ratios = sorted(r["peak_ratio"] for r in rows)
        cut = ratios[int(DROP_BOTTOM * len(ratios))]
        rows = [r for r in rows if r["peak_ratio"] >= cut]
        # Cap the prediction's log hour-1 size at the 90th percentile of the
        # comparables' log hour-1 sizes (see fit_hour).
        xs = sorted(math.log(r["p1"]) for r in rows)
        cap = xs[int(0.9 * len(xs))]
    else:
        cap = None

    # Fallback median curve (all comparables, unweighted) if a fit fails.
    medians = {}
    for h in range(2, 25):
        vals = sorted(r["lr"][h] for r in rows if h in r["lr"])
        medians[h] = vals[len(vals) // 2] if vals else 0.0

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        fit = fit_hour(rows, p1, h, cap) if rows else None
        if fit is None:
            mu, sd = medians[h], SD_CAP
        else:
            mu, sd = fit
        spread = sd * SPREAD_SCALE
        qs = [round2(p1 * math.exp(mu + z * spread)) for z in Z]
        qs = [max(q, 0.01) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
