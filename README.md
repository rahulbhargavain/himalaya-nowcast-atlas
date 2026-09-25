# Himalaya Nowcast Atlas: point-in-time archive

Every build of the [atlas](https://rahulbhargavain.github.io/himalaya-nowcast-atlas/) (every 2 hours) files what it knew at that moment here. Nothing is rewritten afterwards, so any analysis sees only what was available at each build time.

`build_utc` is when the build ran; `valid_utc` / `obs_utc` is the time a value describes.

| File | One row per | Contents |
| --- | --- | --- |
| `builds.csv` | build | radar time, echo area, overall storm motion, cell/station/gauge counts, fire pixels |
| `towns.csv` | build × town | radar dBZ at the town now, extrapolated onset (min), model rain over the next 8 h |
| `forecast.csv` | build × town × hour | model rain and snow line as forecast, with lead time in hours (negative = analysis of past hours) |
| `stations.csv` | airport report | METAR observations (temperature, dew point, wind, QNH, weather, visibility, raw report), each report stored once |
| `gauges.csv` | build × gauge | GloFAS discharge today, % of the 15-year median for the date, and the forecast from today |

`snapshots/YYYY/MM/DD/HHMMZ/` holds the full record of each build:

- `snapshot.json.gz`: everything above plus cells, events, pressure centres, snow line and each town's 10-minute radar series (observed then extrapolated)
- `radar_obs.png`: latest observed radar; `radar_fc_{1,2,4,8}h.png`: the extrapolation for those lead times. Greyscale, pixel value = dBZ (0 = no echo), on a Web Mercator grid at zoom 7 / 4 covering 72–97°E, 26–37°N (origin at zoom-7 pixel x 22936, y 12754, 4 zoom-7 pixels per archive pixel).

Scoring the nowcast: compare `radar_fc_2h.png` from one build with `radar_obs.png` from the build two hours later.

Sources and licences are listed in the main branch README.
