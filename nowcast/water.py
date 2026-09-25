"""Rivers, lakes, river-flow gauges and ground stations.

  * River and lake extents: Natural Earth 1:10m (rivers_lake_centerlines,
    lakes) plus MODIS water pixels from the land cover layer.
  * River flow: GloFAS v4 modelled discharge via the Open-Meteo Flood API,
    compared against the same calendar window in 1995-2024. There is no
    free live gauge-level feed covering the region, so this is modelled
    flow, not a measured stage.
  * Ground stations: airport METARs from aviationweather.gov (NOAA).
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

from .geo import CACHE, E_LON, FRAME, N_LAT, S_LAT, W_LON, get, get_json

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{}.geojson"

# Gauges: approximate main-channel locations. The nearest GloFAS cell with
# the largest mean flow within +-0.15 deg is used, so the point lands on the
# river itself rather than a side stream.
GAUGES = [
    ("Indus", "Leh", 34.13, 77.60),
    ("Indus", "Skardu", 35.33, 75.55),
    ("Jhelum", "Srinagar", 34.08, 74.80),
    ("Chenab", "Akhnoor", 32.89, 74.74),
    ("Sutlej", "Rampur", 31.45, 77.63),
    ("Beas", "Mandi", 31.71, 76.93),
    ("Yamuna", "Dakpathar", 30.50, 77.80),
    ("Ganga", "Rishikesh", 30.10, 78.30),
    ("Alaknanda", "Srinagar (Garhwal)", 30.22, 78.78),
    ("Kali / Sharda", "Tanakpur", 29.07, 80.11),
    ("Karnali", "Chisapani", 28.64, 81.29),
    ("Narayani / Gandaki", "Narayanghat", 27.70, 84.43),
    ("Koshi", "Chatara", 26.87, 87.16),
    ("Teesta", "Sevoke", 26.90, 88.47),
    ("Siang / Brahmaputra", "Pasighat", 28.06, 95.33),
]

# Lakes too small for the 1:10m data, shown as named points.
SMALL_LAKES = [
    ("Bhimtal", 29.345, 79.560), ("Naini Tal", 29.392, 79.455), ("Tehri reservoir", 30.43, 78.43),
    ("Dal Lake", 34.11, 74.87), ("Wular Lake", 34.36, 74.60), ("Tso Moriri", 32.90, 78.31),
    ("Phewa Tal", 28.21, 83.95), ("Rara Lake", 29.53, 82.09), ("Gosaikunda", 28.08, 85.41),
    ("Tsomgo", 27.37, 88.76), ("Pong reservoir", 32.02, 76.07), ("Gobind Sagar", 31.42, 76.50),
]

PLACES = [
    ("Gilgit", 35.92, 74.31), ("Skardu", 35.30, 75.63), ("Srinagar", 34.08, 74.80), ("Leh", 34.16, 77.58),
    ("Jammu", 32.73, 74.86), ("Dharamshala", 32.22, 76.32), ("Manali", 32.24, 77.19), ("Kaza", 32.23, 78.07),
    ("Shimla", 31.10, 77.17), ("Dehradun", 30.32, 78.03), ("Uttarkashi", 30.73, 78.45), ("Joshimath", 30.56, 79.56),
    ("Bhimtal", 29.35, 79.55), ("Pithoragarh", 29.58, 80.21), ("Jumla", 29.27, 82.18), ("Jomsom", 28.78, 83.72),
    ("Pokhara", 28.21, 83.99), ("Kathmandu", 27.71, 85.32), ("Namche Bazaar", 27.80, 86.71),
    ("Darjeeling", 27.04, 88.26), ("Gangtok", 27.33, 88.61), ("Thimphu", 27.47, 89.64), ("Tawang", 27.59, 91.86),
    ("Lhasa", 29.65, 91.17), ("Shigatse", 29.27, 88.88), ("Itanagar", 27.08, 93.61),
]


def _in_box(lon, lat, pad=0.5):
    return W_LON - pad <= lon <= E_LON + pad and S_LAT - pad <= lat <= N_LAT + pad


def _path(coords):
    xs, ys = FRAME.to_grid(np.array([c[0] for c in coords]), np.array([c[1] for c in coords]))
    pts = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)]
    dedup = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if p != a]
    return "M" + "L".join(dedup) if len(dedup) > 1 else ""


def rivers_and_lakes():
    rivers = json.loads(get(NE.format("ne_10m_rivers_lake_centerlines"), cache_key="ne_10m_rivers_lake_centerlines.geojson", timeout=300))
    lakes = json.loads(get(NE.format("ne_10m_lakes"), cache_key="ne_10m_lakes.geojson", timeout=300))
    out_r, out_l = [], []
    for f in rivers["features"]:
        geom = f["geometry"]
        if not geom:
            continue
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        keep = [ln for ln in lines if any(_in_box(*c[:2]) for c in ln)]
        if not keep:
            continue
        d = " ".join(p for p in (_path(ln) for ln in keep) if p)
        if d:
            p = f["properties"]
            out_r.append({"name": p.get("name") or "", "rank": int(p.get("scalerank") or 10),
                          "lake": p.get("featurecla") == "Lake Centerline", "d": d})
    for f in lakes["features"]:
        geom = f["geometry"]
        if not geom:
            continue
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        keep = [pg for pg in polys if any(_in_box(*c[:2], pad=0) for c in pg[0])]
        if not keep:
            continue
        d = " ".join(_path(pg[0]) + "Z" for pg in keep)
        out_l.append({"name": f["properties"].get("name") or "", "d": d})
    small = []
    for name, lat, lon in SMALL_LAKES:
        x, y = FRAME.to_grid(lon, lat)
        small.append({"name": name, "x": float(x), "y": float(y)})
    return out_r, out_l, small


def _flood(lats, lons, **params):
    return get_json("https://flood-api.open-meteo.com/v1/flood", params=dict(
        latitude=",".join(f"{a:.3f}" for a in lats), longitude=",".join(f"{a:.3f}" for a in lons),
        daily="river_discharge", **params), timeout=180)


def _snap_gauges():
    """Pick the main-channel GloFAS cell for each gauge (cached)."""
    path = os.path.join(CACHE, "gauges_snapped.json")
    if os.path.exists(path):
        return json.load(open(path))
    snapped = []
    offs = (-0.15, -0.075, 0.0, 0.075, 0.15)
    for river, place, lat, lon in GAUGES:
        cand = [(lat + a, lon + b) for a in offs for b in offs]
        j = _flood([c[0] for c in cand], [c[1] for c in cand], past_days=30, forecast_days=1)
        j = j if isinstance(j, list) else [j]
        best = max(range(len(j)), key=lambda k: np.nanmean([x for x in j[k]["daily"]["river_discharge"] if x is not None] or [0]))
        snapped.append({"river": river, "place": place, "lat": round(cand[best][0], 3), "lon": round(cand[best][1], 3)})
    json.dump(snapped, open(path, "w"), indent=1)
    return snapped


def gauges():
    g = _snap_gauges()
    lats, lons = [x["lat"] for x in g], [x["lon"] for x in g]
    today = dt.datetime.utcnow().date()
    # Climatology: the same calendar window (60 days back, 10 ahead) in each
    # of the last 15 complete years -- enough for a p10/p50/p90 band without
    # pulling decades of daily data.
    clim_path = os.path.join(CACHE, f"glofas_clim_{today.isoformat()}.json")
    if os.path.exists(clim_path):
        clim_raw = json.load(open(clim_path))
    else:
        clim_raw = [{"daily": {"time": [], "river_discharge": []}} for _ in g]
        for year in range(today.year - 15, today.year):
            try:
                start = today.replace(year=year) - dt.timedelta(days=67)
                end = today.replace(year=year) + dt.timedelta(days=10)
            except ValueError:  # 29 Feb
                continue
            part = _flood(lats, lons, start_date=start.isoformat(), end_date=end.isoformat())
            part = part if isinstance(part, list) else [part]
            for k, pk in enumerate(part):
                clim_raw[k]["daily"]["time"] += pk["daily"]["time"]
                clim_raw[k]["daily"]["river_discharge"] += pk["daily"]["river_discharge"]
        json.dump(clim_raw, open(clim_path, "w"))
    now = _flood(lats, lons, past_days=60, forecast_days=7)
    now = now if isinstance(now, list) else [now]
    out = []
    for k, meta in enumerate(g):
        ct = np.array(clim_raw[k]["daily"]["time"], dtype="datetime64[D]")
        cv = np.array([np.nan if v is None else v for v in clim_raw[k]["daily"]["river_discharge"]], float)
        doy_c = (ct - ct.astype("datetime64[Y]")).astype(int)
        times = now[k]["daily"]["time"]
        vals = [None if v is None else float(v) for v in now[k]["daily"]["river_discharge"]]
        band = []
        for t in times:
            d = np.datetime64(t, "D")
            doy = int((d - d.astype("datetime64[Y]")).astype(int))
            sel = np.abs(((doy_c - doy + 183) % 366) - 183) <= 7
            vv = cv[sel & np.isfinite(cv)]
            band.append([round(float(np.percentile(vv, p)), 1) for p in (10, 50, 90)] if vv.size else [None] * 3)
        i_today = times.index(today.isoformat()) if today.isoformat() in times else len(times) - 8
        cur = vals[i_today]
        med = band[i_today][1]
        x, y = FRAME.to_grid(meta["lon"], meta["lat"])
        out.append({**meta, "x": float(x), "y": float(y), "times": times, "q": vals, "band": band,
                    "today_index": i_today, "q_now": cur,
                    "pct_of_median": round(100 * cur / med) if cur is not None and med else None})
    return out


def metars():
    j = get_json("https://aviationweather.gov/api/data/metar",
                 params=dict(bbox=f"{S_LAT},{W_LON},{N_LAT},{E_LON}", format="json", hours=3), timeout=60) or []
    latest = {}
    for m in j:
        if m.get("icaoId") and (m["icaoId"] not in latest or m["obsTime"] > latest[m["icaoId"]]["obsTime"]):
            latest[m["icaoId"]] = m
    out = []
    for m in latest.values():
        x, y = FRAME.to_grid(m["lon"], m["lat"])
        if not (0 <= x < FRAME.gw and 0 <= y < FRAME.gh):
            continue
        out.append({
            "id": m["icaoId"], "name": (m.get("name") or "").split(",")[0], "x": float(x), "y": float(y),
            "elev": m.get("elev"), "t": m.get("temp"), "td": m.get("dewp"), "wdir": m.get("wdir"),
            "wspd": m.get("wspd"), "qnh": m.get("altim"), "wx": m.get("wxString") or "",
            "vis": m.get("visib"), "cover": m.get("cover"), "obs": m.get("obsTime"), "raw": m.get("rawOb"),
        })
    out.sort(key=lambda s: s["x"])
    return out


def places():
    out = []
    for name, lat, lon in PLACES:
        x, y = FRAME.to_grid(lon, lat)
        out.append({"name": name, "lat": lat, "lon": lon, "x": float(x), "y": float(y)})
    return out
