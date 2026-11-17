# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arshamax16/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arshamax16/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun's **"Forecast GTA VI's launch-day Twitch viewers hour by hour"** bounty ($250 USDC). Original Python program (stdlib only, no network, deterministic) that forecasts hours 2–24 of GTA VI's launch-day Twitch CCU from the hour-1 print and the launch-day shapes of 42 past big launches.

## Method

A launch day on Twitch is a rolling midnight unlock: hour 1 only has the regions that already unlocked, the US unlocks lift the morning, and the curve climbs to an evening peak. The program learns that shape from the case's own comparables instead of assuming it:

- For each forecast hour, it regresses the log-ratio to hour 1 on log hour-1 size, weighting past launches by how close their hour-1 size is to this case's (Gaussian kernel, bandwidth 1.5 in log units).
- It drops the weakest 15% of comparables by peak ratio (flops and thin categories whose hour-1 print was barely a launch).
- Quantile spreads come from that hour's regression residuals, capped at 0.6 log units and scaled by 0.65.
- The prediction's log hour-1 size is capped at the 90th percentile of the comparables — the only two comparables bigger than that had simultaneous global unlocks and peaked in hour 1, a pattern a rolling-midnight console launch will not repeat.

The key empirical fact: across comparables the evening peak is almost independent of the hour-1 print (log peak ≈ 11.7 + 0.07 · log p1), so a small hour-1 print means a much bigger rise ahead. A flat persistence forecast (the house baseline) misses that entirely.

## Results on the practice cases

Mean pinball loss on the log scale (lower is better):

| case | this entry | house baseline |
|------|-----------|----------------|
| p00 Wolverine | 0.144 | 0.744 |
| p01 Red Dead Redemption 2 | 0.190 | 0.287 |
| p02 God of War Ragnarök | 0.109 | 0.215 |
| p03 Spider-Man 2 | 0.170 | 0.303 |
| **mean** | **0.153** | **0.387** |

Two runs on the same input produce byte-identical outputs; the whole 4-case practice set runs in ~0.3 s.

## Usage

```sh
python3 -P -s main.py   # reads $MAKEFUN_INPUT_DIR (else /input/), writes $MAKEFUN_OUTPUT_DIR (else /output/)
```

Each case writes `<id>.out`: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2–24.

## License

MIT — see [LICENSE](LICENSE).
