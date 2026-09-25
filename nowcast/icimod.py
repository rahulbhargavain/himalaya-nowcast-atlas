"""ICIMOD (International Centre for Integrated Mountain Development) layers
for the Hindu Kush Himalaya, from its open ArcGIS REST services at
geoapps.icimod.org, plus live VIIRS fire detections from NASA GIBS.

  * Thematic rasters and polygon layers are fetched with MapServer/export
    straight onto the atlas's Web Mercator grid, in ICIMOD's own symbology,
    with the matching legend swatches.
  * Year series (RLCMS land cover, glacier outlines, population,
    temperature anomaly) feed the page's "years" slider.
  * Point inventories (potentially dangerous glacial lakes, GLOF events,
    hydropower plants, landslides, meteorological stations) are queried as
    features and shown as markers.

Deliberately NOT used: the Bhutan "Pastoral Migration" household points,
which carry names, national ID numbers and phone numbers. Only the
grazing-area and tsamdro (pasture) maps from that service are shown.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json

import numpy as np
from PIL import Image

from .geo import E_LON, FRAME, N_LAT, S_LAT, W_LON, get, get_json

ROOT = "https://geoapps.icimod.org/icimodarcgis/rest/services"
MONTH_S = 30 * 86400

# key, group, title, service, layer ids, one-line description
THEMES = [
    ("pop2025", "people", "Population 2025", "RIS/HKH_Demography", "0", "People per km², WorldPop-based estimate for 2025."),
    ("sexratio", "people", "Sex ratio 2025", "RIS/HKH_Demography", "1", "Males per 100 females. High values often mark male labour migration into an area; low values, out-migration."),
    ("childwoman", "people", "Child–woman ratio 2025", "RIS/HKH_Demography", "3", "Children under 5 per woman of child-bearing age, a fertility indicator."),
    ("aged", "people", "Share aged 75+", "RIS/HKH_Demography", "2", "Proportion of the population aged 75 and over."),
    ("deprivation", "people", "Relative deprivation", "RIS/HKH_HumanDimensions", "1", "Relative Deprivation Index (income, education, health, living standards)."),
    ("impervious", "people", "Built-up surface", "RIS/HKH_HumanDimensions", "0", "Impervious (built) surface: a proxy for urban versus rural habitat."),
    ("nightlight", "people", "Night lights 2015–2025", "RIS/HKH_Demography", "7", "VIIRS night-time lights, showing settlement and electrification."),
    ("roads", "people", "Roads, rail, airfields", "HKH/Infrastructure", "0,4,5", "Road, railway and airfield network."),
    ("conversion", "land", "Land conversion pressure", "RIS/HKH_Ecosystem", "0", "Conversion Pressure Index: how exposed natural land is to conversion."),
    ("protected", "land", "Protected areas & bird areas", "HKH/Environment", "0,1", "Protected areas and Important Bird Areas."),
    ("grazing", "land", "Bhutan grazing & tsamdro", "Bhutan/PastoralMigration", "0,2", "Yak and cattle grazing areas and registered tsamdro (pasture) in Bhutan."),
    ("lsusc", "hazard", "Landslide susceptibility", "RIS/HKH_Ecosystem", "1", "Rainfall-triggered landslide susceptibility."),
    ("firecount", "hazard", "Fire count 2001–2025", "HKH/ForestFire", "1", "MODIS active-fire detections per cell, 2001–2025."),
    ("water2023", "water", "Permanent water 2023", "ScienceApps/PermanentWaterbody", "0", "Permanent water bodies mapped from 2023 imagery."),
    ("snowtrend", "climate", "Snowfall trend", "RIS/HKH_Snowfall", "0", "Annual snowfall trend per decade."),
    ("ttrend", "climate", "Temperature trend", "RIS/HKH_Temperature_Trend_Decadal", "0", "Annual temperature trend per decade."),
    ("ptrend", "climate", "Precipitation trend", "RIS/HKH_Precipitation_Trend_Decadal", "0", "Annual precipitation trend, mm per decade."),
]

# Year series for the "years" slider: key -> (title, service, {year: layer id}, description)
SERIES = {
    "landcover": ("Land cover (RLCMS)", "HKH/Landcover",
                  {2000: 0, 2005: 5, 2010: 10, 2015: 14, 2020: 19, 2022: 21},
                  "ICIMOD Regional Land Cover Monitoring System, 30 m, harmonised classes for the whole HKH."),
    "glacier": ("Glacier extent", "HKH/Glacier_1990_2020", {1990: 3, 2000: 2, 2010: 1, 2020: 0},
                "Glacier outlines mapped from Landsat for each decade."),
    "population": ("Population", "HKH/HKHWorldPopPopulation", {2015: 3, 2020: 2, 2025: 1, 2030: 0},
                   "WorldPop population, 2015–2025 and a 2030 projection."),
    "tanomaly": ("Temperature anomaly", "RIS/HKH_Temperature_Anomaly", None,
                 "Annual mean temperature against the long-term average."),
}


def _bbox():
    minx, miny, maxx, maxy = FRAME.merc_bbox()
    return f"{minx},{miny},{maxx},{maxy}"


def export(service, layers, cache_days=30):
    """ICIMOD MapServer export on the atlas GRID, as an RGBA array (or None)."""
    params = dict(bbox=_bbox(), bboxSR=3857, imageSR=3857, size=f"{FRAME.gw},{FRAME.gh}",
                  format="png32", transparent="true", layers=f"show:{layers}", f="image", dpi=96)
    key = f"icimod/export_{service.replace('/', '_')}_{layers.replace(',', '-')}.png"
    body = get(f"{ROOT}/{service}/MapServer/export", params=params, cache_key=key,
               max_age=cache_days * 86400, timeout=180)
    if not body:
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(body)).convert("RGBA"))
    except Exception:
        return None


def legend(service, layer_ids):
    """Legend swatches for the given layer ids: [{label, img}] (img = data URI)."""
    j = get_json(f"{ROOT}/{service}/MapServer/legend", params={"f": "json"},
                 cache_key=f"icimod/legend_{service.replace('/', '_')}.json", max_age=MONTH_S)
    ids = {int(x) for x in str(layer_ids).split(",")}
    out = []
    for L in (j or {}).get("layers", []):
        if L.get("layerId") not in ids:
            continue
        for e in L.get("legend", [])[:14]:
            if e.get("imageData"):
                out.append({"label": e.get("label") or L.get("layerName", ""),
                            "img": f"data:{e.get('contentType', 'image/png')};base64,{e['imageData']}"})
    return out


def _png(arr):
    im = Image.fromarray(arr, "RGBA")
    if (arr[..., 3] > 0).mean() > 0.02:
        im = im.quantize(colors=96, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _anomaly_years(service):
    j = get_json(f"{ROOT}/{service}/MapServer", params={"f": "json"},
                 cache_key=f"icimod/svc_{service.replace('/', '_')}.json", max_age=MONTH_S)
    years = {}
    for L in (j or {}).get("layers", []):
        tail = L["name"].split()[-1]
        if tail.isdigit():
            years[int(tail)] = L["id"]
    # Every 5 years plus the latest, to keep the page light.
    keep = sorted(y for y in years if y % 5 == 0) + [max(years)] if years else []
    return {y: years[y] for y in sorted(set(keep))}


def themes():
    out = []
    for key, group, title, svc, lay, desc in THEMES:
        arr = export(svc, lay)
        if arr is None or not (arr[..., 3] > 0).any():
            continue
        out.append({"key": key, "group": group, "title": title, "desc": desc, "src": _png(arr),
                    "legend": legend(svc, lay), "service": svc})
    return out


def series():
    out = {}
    for key, (title, svc, years, desc) in SERIES.items():
        if years is None:
            years = _anomaly_years(svc)
        frames = []
        for year, lay in years.items():
            arr = export(svc, str(lay))
            if arr is not None and (arr[..., 3] > 0).any():
                frames.append({"year": year, "src": _png(arr)})
        if frames:
            first = next(iter(years.values()))
            out[key] = {"title": title, "desc": desc, "frames": frames, "legend": legend(svc, first), "service": svc}
    return out


def _query_all(service, layer, fields, where="1=1", spatial=True):
    """All features (within the atlas box when `spatial`), paging past the
    server's record cap."""
    feats, offset = [], 0
    env = json.dumps({"xmin": W_LON, "ymin": S_LAT, "xmax": E_LON, "ymax": N_LAT, "spatialReference": {"wkid": 4326}})
    geo = dict(geometry=env, geometryType="esriGeometryEnvelope", inSR=4326, spatialRel="esriSpatialRelIntersects") if spatial else {}
    while True:
        j = get_json(f"{ROOT}/{service}/MapServer/{layer}/query", params=dict(
            where=where, **geo,
            outFields=",".join(fields), returnGeometry="true", outSR=4326, f="json",
            # Small layers without pagination support are fetched in one go.
            **(dict(resultOffset=offset, resultRecordCount=1000) if spatial else {})),
            cache_key=f"icimod/q_{service.replace('/', '_')}_{layer}_{offset}.json", max_age=MONTH_S, timeout=180)
        page = (j or {}).get("features", [])
        feats += page
        if not spatial or len(page) < 1000 or not (j or {}).get("exceededTransferLimit", len(page) == 1000):
            break
        offset += 1000
    return feats


def _xy(f, lat_field=None, lon_field=None):
    g = f.get("geometry") or {}
    a = f["attributes"]
    if "x" in g:
        lon, lat = g["x"], g["y"]
    elif lat_field and a.get(lat_field) is not None:
        lat, lon = a[lat_field], a[lon_field]
    elif "rings" in g:
        ring = np.array(g["rings"][0])
        lon, lat = ring[:, 0].mean(), ring[:, 1].mean()
    else:
        return None
    x, y = FRAME.to_grid(lon, lat)
    if not (0 <= x < FRAME.gw and 0 <= y < FRAME.gh):
        return None
    return float(x), float(y), float(lat), float(lon)


def points():
    out = {}
    # Potentially dangerous glacial lakes (2015 inventory).
    pdgl = []
    # Only 47 lakes; this layer's spatial filter misbehaves, so fetch all and filter by position.
    for f in _query_all("HKH/GlacialLake", 2, ["GL_ID", "Latitude", "Longitude", "Basin", "Sub_Basin", "Area", "Elevation", "Type", "Rank", "Country"], spatial=False):
        p = _xy(f, "Latitude", "Longitude")
        if p:
            a = f["attributes"]
            pdgl.append({"x": p[0], "y": p[1], "id": a.get("GL_ID"), "basin": a.get("Sub_Basin") or a.get("Basin"),
                         "area": a.get("Area"), "elev": a.get("Elevation"), "type": a.get("Type"), "rank": a.get("Rank"), "country": a.get("Country")})
    out["pdgl"] = pdgl
    # Historical GLOF events.
    glof = []
    for f in _query_all("HKH/GLOF", 0, ["Year_exact", "Year_approx", "Lake_name", "Glacier_name", "Impact_type", "Lake_type",
                                         "Country", "River_Basin", "Driver_GLOF", "Lat_lake", "Lon_lake"]):
        p = _xy(f, "Lat_lake", "Lon_lake")
        if p:
            a = f["attributes"]
            yr = a.get("Year_exact") if isinstance(a.get("Year_exact"), (int, float)) else a.get("Year_approx")
            glof.append({"x": p[0], "y": p[1], "year": int(yr) if isinstance(yr, (int, float)) and yr > 0 else None,
                         "lake": a.get("Lake_name"), "glacier": a.get("Glacier_name"), "impact": a.get("Impact_type"),
                         "type": a.get("Lake_type"), "country": a.get("Country"), "basin": a.get("River_Basin"), "driver": a.get("Driver_GLOF")})
    out["glof"] = glof
    # Hydropower plants by status.
    hydro = []
    for lay, status in ((0, "operational"), (1, "under construction"), (2, "planned")):
        for f in _query_all("HKH/HydropowerPlant", lay, ["ProjectName", "CapacityMW", "River", "Country", "StartYear"]):
            p = _xy(f)
            if p:
                a = f["attributes"]
                hydro.append({"x": p[0], "y": p[1], "name": a.get("ProjectName"), "mw": a.get("CapacityMW"),
                              "river": a.get("River"), "country": a.get("Country"), "year": a.get("StartYear"), "status": status})
    out["hydro"] = hydro
    # Landslide events (NASA Global Landslide Catalog, via ICIMOD).
    ls = []
    for f in _query_all("HKH/Landslide", 0, ["ev_date", "ev_title", "ls_trig", "ls_size", "fatalities", "ctry_name"]):
        p = _xy(f)
        if p:
            a = f["attributes"]
            d = a.get("ev_date")
            ls.append({"x": p[0], "y": p[1], "date": dt.datetime.fromtimestamp(d / 1000, dt.timezone.utc).date().isoformat() if d else None,
                       "title": a.get("ev_title"), "trigger": a.get("ls_trig"), "size": a.get("ls_size"),
                       "deaths": a.get("fatalities"), "country": a.get("ctry_name")})
    out["landslides"] = ls
    # Meteorological station network (locations; not live data).
    met = []
    for f in _query_all("HKH/MetStation", 0, ["STN_NAME", "STN_TYPE", "COUNTRY", "SOURCE", "ELEVATION", "ACCESSBILI"]):
        p = _xy(f)
        if p:
            a = f["attributes"]
            met.append({"x": p[0], "y": p[1], "name": a.get("STN_NAME"), "type": a.get("STN_TYPE"), "country": a.get("COUNTRY"),
                        "source": a.get("SOURCE"), "elev": a.get("ELEVATION"), "access": a.get("ACCESSBILI")})
    out["met"] = met
    return out


def live_fires(days=2):
    """VIIRS (Suomi NPP + NOAA-20) thermal anomalies for the last `days`
    days, as an RGBA overlay from NASA GIBS, plus the dates used."""
    from .land import GIBS_WMS
    minx, miny, maxx, maxy = FRAME.merc_bbox()
    acc = np.zeros((FRAME.gh, FRAME.gw, 4), np.uint8)
    used = []
    today = dt.datetime.now(dt.timezone.utc).date()
    for back in range(0, days + 1):
        d = today - dt.timedelta(days=back)
        for layer in ("VIIRS_SNPP_Thermal_Anomalies_375m_All", "VIIRS_NOAA20_Thermal_Anomalies_375m_All"):
            body = get(GIBS_WMS, params=dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS=layer, STYLES="",
                                             CRS="EPSG:3857", BBOX=f"{minx},{miny},{maxx},{maxy}", WIDTH=FRAME.gw, HEIGHT=FRAME.gh,
                                             FORMAT="image/png", TRANSPARENT="TRUE", TIME=d.isoformat()), timeout=120)
            if not body:
                continue
            a = np.asarray(Image.open(io.BytesIO(body)).convert("RGBA"))
            hit = a[..., 3] > 0
            if hit.any():
                acc[hit] = [255, 80, 20, 255]
                if d not in used:
                    used.append(d)
    n = int((acc[..., 3] > 0).sum())
    return {"src": _png(acc) if n else None, "pixels": n, "dates": [d.isoformat() for d in sorted(used)]}
