# Himalaya Nowcast Atlas

A single-page atlas of weather and seasons over the Himalaya, Kashmir to Arunachal (72–97°E, 26–37°N).

**Live page:** https://rahulbhargavain.github.io/himalaya-nowcast-atlas/ (rebuilt every 2 hours)

## What it shows

**Timeline (−2 h to +8 h, 10-minute steps)**
- Radar: the last two hours of RainViewer composite radar, then an 8-hour extrapolation using optical flow.
- Model rain, MSL pressure (isobars, highs and lows) and the hourly snow line from the Open-Meteo forecast.
- Timeline events: when echo reaches a town, when model rain starts, when pressure centres form, and snow line shifts.

**Surface**
- Shaded relief, MODIS land cover, and the tree line and shrub line drawn from the land-cover edges.
- Snow cover over the last 8 days (MODIS).
- Rivers and lakes (Natural Earth), and 15 river gauges comparing modelled flow (GloFAS) with the same days over the last 15 years.
- Ground truth: METARs from airports in and near the mountains.

**Seasons (month selector)**
- Typical snow cover for the month and the seasonal snow line per sector (MODIS monthly averages, last 3 years).
- Snow leopard habitat band by month (modelled from elevation, slope and land cover).
- Fruiting belts: kafal, hisalu, timla, apricot, apple, large cardamom, walnut, seabuckthorn, mandarin.

**Hindu Kush Himalaya data (ICIMOD)**
- Thematic layers from ICIMOD's open map services: population, sex ratio, child–woman ratio, share aged 75+, relative deprivation, built-up surface (urban/rural), night lights, roads, land conversion pressure, protected areas, Bhutan grazing areas, landslide susceptibility, fire history, permanent water, and snowfall, temperature and precipitation trends.
- Change over the years: RLCMS land cover 2000–2022, glacier extent 1990–2020, population 2015–2030, temperature anomaly since 1995.
- Inventories: potentially dangerous glacial lakes, GLOF events since the 1500s, hydropower plants (operating, building, planned), landslides with fatalities, and the meteorological station network.
- Live fires: VIIRS thermal anomalies for the last 48 hours (NASA GIBS).

## How it works

`build.py` fetches everything, computes the layers and writes `dist/index.html`, one self-contained page with all data embedded:

| Module | Does |
| --- | --- |
| `nowcast/radar.py` | RainViewer frames → dBZ → OpenCV DIS optical flow (blended with 500/700 hPa steering wind where there is no echo) → semi-Lagrangian extrapolation |
| `nowcast/model.py` | Open-Meteo grid: pressure contours and H/L centres, snow line (freezing level − 300 m) contoured on the terrain, steering wind |
| `nowcast/land.py` | Terrain tiles, hillshade, MODIS land cover and snow via NASA GIBS, tree line edges, seasonal snow line |
| `nowcast/water.py` | Natural Earth rivers and lakes, GloFAS gauges, METARs, places |
| `nowcast/icimod.py` | ICIMOD ArcGIS REST layers (export images, legends, feature inventories) and VIIRS fires |

Radar extrapolation assumes storms keep their motion and strength. It is useful for about the first two hours; after that, compare it with the model rain layer.

## Point-in-time archive

Each build also writes a compact record of what it knew (town-by-town radar and model rain, snow line, pressure centres, airport observations, river flow, and the observed radar with its +1/+2/+4/+8 h extrapolation). The workflow appends it to the [`data` branch](https://github.com/rahulbhargavain/himalaya-nowcast-atlas/tree/data), about 30 KB per build. Nothing already archived is rewritten, so analyses see only what was known at the time. See that branch's README for the tables.

## TV mode

Add `?tv` to the URL (https://rahulbhargavain.github.io/himalaya-nowcast-atlas/?tv) for a kiosk view: just the map, legend and time, with a fixed set of weather layers, looping through the timeline. The [weather-android-tv](https://github.com/rahulbhargavain/weather-android-tv) dashboard embeds this view.

## Run locally

```bash
pip install -r requirements.txt
python build.py
```

Open `dist/index.html`. Static layers are cached in `cache/` after the first run (about 1 minute); later builds take about 30 seconds.

## Data sources

All free and keyless: [RainViewer](https://www.rainviewer.com/api.html), [Open-Meteo](https://open-meteo.com/) (forecast and flood APIs, non-commercial use), [NASA GIBS](https://nasa-gibs.github.io/gibs-api-docs/) (MODIS), [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/), [Natural Earth](https://www.naturalearthdata.com/), [aviationweather.gov](https://aviationweather.gov/data/api/), [ICIMOD](https://geoapps.icimod.org/icimodarcgis/rest/services) (Regional Database System and SERVIR-HKH).

ICIMOD's Bhutan pastoral migration survey points include household names and ID numbers, so the atlas uses only the grazing-area maps from that service.
