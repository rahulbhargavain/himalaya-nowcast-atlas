"""Build the Himalaya weather atlas as one self-contained HTML page.

    python build.py                 # writes dist/index.html
    python build.py --out some.html

Static layers (terrain, land cover, monthly snow, rivers) are cached in
./cache and fetched once. Radar, model, river flow and stations are
fetched fresh on every run, so re-running refreshes the page.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

from nowcast import icimod, land, model, radar, water
from nowcast.geo import FRAME, world_lonlat

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def png_uri(arr, mode=None, quantize=None):
    im = Image.fromarray(arr, mode) if mode else Image.fromarray(arr)
    if quantize:
        im = im.quantize(colors=quantize, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def jpg_uri(arr, q=84):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Fruiting calendar and snow leopard band. Elevation ranges and months are
# approximate, compiled from regional horticulture and ecology literature;
# they vary with aspect, local climate and year.
FRUITS = [
    {"id": "kafal", "name": "Kafal", "latin": "Myrica esculenta", "months": [4, 5, 6], "elev": [900, 2100], "lon": [74, 92], "lc": "wild"},
    {"id": "hisalu", "name": "Hisalu (golden raspberry)", "latin": "Rubus ellipticus", "months": [4, 5, 6], "elev": [700, 2300], "lon": [73, 96], "lc": "wild"},
    {"id": "fig", "name": "Timla (Himalayan fig)", "latin": "Ficus auriculata", "months": [5, 6, 7], "elev": [600, 1700], "lon": [74, 95], "lc": "wild"},
    {"id": "apricot", "name": "Apricot", "latin": "Prunus armeniaca", "months": [6, 7, 8], "elev": [1800, 3600], "lon": [73, 85], "lc": "orchard_dry"},
    {"id": "apple", "name": "Apple", "latin": "Malus domestica", "months": [8, 9, 10], "elev": [1500, 2800], "lon": [73, 84], "lc": "orchard"},
    {"id": "cardamom", "name": "Large cardamom (harvest)", "latin": "Amomum subulatum", "months": [8, 9, 10], "elev": [600, 2000], "lon": [85.5, 93], "lc": "wild"},
    {"id": "walnut", "name": "Walnut", "latin": "Juglans regia", "months": [9, 10], "elev": [1200, 2800], "lon": [73, 90], "lc": "orchard"},
    {"id": "seabuckthorn", "name": "Seabuckthorn", "latin": "Hippophae rhamnoides", "months": [9, 10], "elev": [2500, 4300], "lon": [74, 86], "lc": "orchard_dry"},
    {"id": "orange", "name": "Mandarin orange", "latin": "Citrus reticulata", "months": [11, 12, 1], "elev": [500, 1500], "lon": [84, 93], "lc": "orchard"},
]
# Snow leopard: lower and upper elevation of the typical habitat band by
# month (index 0 = January). They follow wild sheep and ibex up to alpine
# pastures in summer and down toward the treeline in winter.
LEOPARD = {
    "lo": [2500, 2550, 2750, 3050, 3400, 3650, 3700, 3650, 3400, 3050, 2750, 2550],
    "hi": [4600, 4650, 4800, 5050, 5350, 5550, 5600, 5550, 5350, 5050, 4800, 4650],
    "min_slope": 12,
}


def town_timeline(places, frames_all, t_all, t0_index, g, t0):
    out = []
    for p in places:
        ix, iy = int(round(p["x"])), int(round(p["y"]))
        if not (2 <= ix < FRAME.gw - 2 and 2 <= iy < FRAME.gh - 2):
            continue
        dbz = [float(f[iy - 3:iy + 4, ix - 3:ix + 4].max()) for f in frames_all]
        now_dbz = dbz[t0_index]
        onset = None
        for k in range(t0_index + 1, len(dbz)):
            if dbz[k] >= 25:
                onset = int((t_all[k] - t0) / 60)
                break
        ps = model.point_series(g, p["lon"], p["lat"])
        mt = g["times"]
        rain_model = [None if not np.isfinite(v) else round(float(v), 1) for v in ps["precipitation"]]
        sl = [None if not np.isfinite(v) else int(v - model.SNOWLINE_OFFSET_M) for v in ps["freezing_level_height"]]
        future = [(t, r) for t, r in zip(mt, rain_model) if t > t0 and r is not None]
        out.append({
            "name": p["name"], "x": p["x"], "y": p["y"],
            "dbz_series": [round(v) for v in dbz], "dbz_now": round(now_dbz),
            "radar_onset_min": onset if now_dbz < 25 else 0,
            "model_rain_8h": round(sum(r for _, r in future), 1),
            "model_rain": rain_model, "snowline": sl,
        })
    return out


def events(towns, pressure, g, t0):
    """Timeline events within the next 8 h, grouped so the scrubber stays readable."""
    ev = []
    radar_on = {}
    for tw in towns:
        if tw["radar_onset_min"]:
            radar_on.setdefault(tw["radar_onset_min"], []).append(tw["name"])
    for t, names in radar_on.items():
        ev.append({"t": t, "kind": "radar", "text": "Extrapolated echo reaches " + ", ".join(names)})
    rain_on = {}
    for tw in towns:
        for t, r in zip(g["times"], tw["model_rain"]):
            if t > t0 and r is not None and r >= 1.0:
                rain_on.setdefault(int((t - t0) / 60), []).append(f"{tw['name']} {r:g}")
                break
    for t, items in rain_on.items():
        ev.append({"t": t, "kind": "model", "text": "Model rain from " + ", ".join(items) + " mm/h"})
    for tw in towns:
        sl = [(t, s) for t, s in zip(g["times"], tw["snowline"]) if s is not None and t >= t0]
        if len(sl) > 2:
            first = sl[0][1]
            for t, s in sl[1:]:
                if abs(s - first) >= 300:
                    ev.append({"t": int((t - t0) / 60), "kind": "snow",
                               "text": f"Snow line {'drops' if s < first else 'rises'} to {s:,} m over {tw['name']}"})
                    break
    # Pressure centres: only ones that appear after the current hour.
    def key(c):
        return (c["kind"], round(c["lon"] / 1.5), round(c["lat"] / 1.5))
    seen = set()
    for t, hour in zip(g["times"], pressure):
        for c in hour["centres"]:
            if t <= t0 + 1800:
                seen.add(key(c))
            elif key(c) not in seen:
                seen.add(key(c))
                ev.append({"t": int((t - t0) / 60), "kind": "pressure",
                           "text": f"{'High' if c['kind'] == 'H' else 'Low'} {c['hpa']:.0f} hPa appears near {c['lat']:.1f}N {c['lon']:.1f}E"})
    ev.sort(key=lambda e: e["t"])
    return [e for e in ev if 0 < e["t"] <= 480][:30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "dist", "index.html"))
    args = ap.parse_args()

    lon_g, lat_g = FRAME.grid_lonlat()
    lat_rows_full = world_lonlat(np.full(FRAME.h, FRAME.x0), FRAME.y0 + np.arange(FRAME.h) + 0.5)[1]

    log("terrain")
    dem = land.load_dem()
    dem_g = cv2.resize(dem, (FRAME.gw, FRAME.gh), interpolation=cv2.INTER_AREA)
    slope_g = land.slope_deg(dem_g, lat_g[:, 0], scale=2)

    log("land cover")
    lc, lc_year = land.land_cover()
    water_full = cv2.resize((lc == 17).astype(np.uint8), (FRAME.w, FRAME.h), interpolation=cv2.INTER_NEAREST).astype(bool)
    shade = land.hillshade_image(dem, lat_rows_full, water_full)
    tree_edge, shrub_edge = land.treeline(dem_g, lc, lon_g, lat_g)
    tree_stats = land.sector_stats(tree_edge, dem_g, lon_g, lat_g, pct=90)
    shrub_stats = land.sector_stats(shrub_edge, dem_g, lon_g, lat_g, pct=90)

    log("snow (recent days, monthly)")
    snow_recent, snow_dates = land.recent_snow()
    snow_month, snow_years = land.monthly_snow()
    season_line = land.seasonal_snowline(snow_month, dem_g, lon_g, lat_g)

    log("model grid")
    g = model.fetch_grid()

    log("radar frames")
    r_times, stack, covered = radar.fetch_frames()
    t0 = r_times[-1]
    t_model = int(np.argmin([abs(t - t0) for t in g["times"]]))
    steer = model.steering_uv(g, t_model)
    log("optical flow + extrapolation")
    v, conf = radar.motion_field(stack, steer)
    fc = radar.extrapolate(stack[-1], v)
    frames_all = list(stack) + fc
    t_all = list(r_times) + [t0 + 60 * radar.STEP_MIN * (k + 1) for k in range(len(fc))]
    t0_index = len(r_times) - 1

    places = water.places()
    cells = radar.cells(stack[-1], v, lon_g, lat_g, places)
    echo = stack[-1] > radar.MIN_DBZ
    if echo.sum() > 50:
        mv = np.median(v[echo], axis=0)
        motion = {"speed_kmh": round(float(np.hypot(*mv)) * 2.1 * 6),
                  "heading": round(float((np.degrees(np.arctan2(mv[0], -mv[1])) + 360) % 360)),
                  "echo_km2": int(echo.sum() * 2.1 ** 2)}
    else:
        motion = None

    log("pressure + snow line layers")
    pressure = model.pressure_layers(g, dem_g)
    snowline = model.snowline_layers(g, dem_g)

    log("rivers, lakes, gauges, stations")
    rivers, lakes, small_lakes = water.rivers_and_lakes()
    try:
        gauges = water.gauges()
    except Exception as e:  # river flow is a nice-to-have; don't lose the page
        log(f"  gauges failed: {e}")
        gauges = []
    try:
        stations = water.metars()
    except Exception as e:
        log(f"  METARs failed: {e}")
        stations = []

    # Keep airports in or next to the mountains (terrain over 1,500 m within ~60 km).
    near_hi = cv2.dilate((dem_g > 1500).astype(np.uint8), np.ones((57, 57), np.uint8)).astype(bool)
    stations = [st for st in stations
                if 0 <= int(st["y"]) < FRAME.gh and 0 <= int(st["x"]) < FRAME.gw and near_hi[int(st["y"]), int(st["x"])]]

    log("ICIMOD layers")
    hkh = {}
    for key, fn in (("themes", icimod.themes), ("series", icimod.series), ("points", icimod.points), ("fires", icimod.live_fires)):
        try:
            hkh[key] = fn()
        except Exception as e:  # one unavailable service shouldn't sink the page
            log(f"  ICIMOD {key} failed: {e}")

    towns = town_timeline(places, frames_all, t_all, t0_index, g, t0)
    evs = events(towns, pressure, g, t0)

    log("encoding")
    elev_u16 = np.clip(dem_g + 500, 0, 65535).astype(np.uint16)
    terrain_png = png_uri(np.dstack([(elev_u16 >> 8).astype(np.uint8), (elev_u16 & 255).astype(np.uint8), lc]), "RGB")
    flags = (tree_edge.astype(np.uint8) * 1) | (shrub_edge.astype(np.uint8) * 2) | (covered.astype(np.uint8) * 4)
    aux_png = png_uri(np.dstack([np.clip(slope_g, 0, 90).astype(np.uint8), np.clip(snow_recent, 0, 100).astype(np.uint8), flags]), "RGB")
    month_pngs = [png_uri(np.dstack([snow_month[i], snow_month[i + 1], snow_month[i + 2]]), "RGB") for i in range(0, 12, 3)]
    radar_frames = []
    for k, (t, f) in enumerate(zip(t_all, frames_all)):
        radar_frames.append({"t": int(t), "obs": k <= t0_index, "src": png_uri(radar.colorize(f), "RGBA", quantize=48)})

    precip = np.nan_to_num(g["precipitation"], nan=0)
    data = {
        "generated": int(time.time()), "t0": int(t0), "step_min": radar.STEP_MIN,
        "frame": {"w": FRAME.w, "h": FRAME.h, "gw": FRAME.gw, "gh": FRAME.gh,
                  "x0": FRAME.x0, "y0": FRAME.y0, "bbox": [72, 26, 97, 37]},
        "images": {"shade": jpg_uri(shade), "terrain": terrain_png, "aux": aux_png, "snow_month": month_pngs},
        "radar": {"frames": radar_frames, "t0_index": t0_index, "cells": cells, "motion": motion,
                  "dbz_stops": radar.DBZ_STOPS.tolist(), "dbz_rgba": radar.DBZ_RGBA.tolist()},
        "model": {"times": [int(t) for t in g["times"]], "lats": g["lats"].tolist(), "lons": g["lons"].tolist(),
                  "precip10": np.round(precip * 10).astype(int).tolist(),
                  "freeze10": np.round(np.nan_to_num(g["freezing_level_height"], nan=0) / 10).astype(int).tolist(),
                  "pressure": pressure, "snowline": snowline, "snow_offset": model.SNOWLINE_OFFSET_M},
        "land": {"lc_year": lc_year, "igbp": land.IGBP, "tree": tree_stats, "shrub": shrub_stats,
                 "snow_dates": [str(d) if d else None for d in snow_dates], "snow_years": snow_years,
                 "season_snowline": season_line, "sectors": land.SECTORS},
        "water": {"rivers": rivers, "lakes": lakes, "small_lakes": small_lakes, "gauges": gauges},
        "stations": stations, "places": places, "towns": towns, "events": evs,
        "fruits": FRUITS, "leopard": LEOPARD, "hkh": hkh,
    }

    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    body = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    # index.html: a complete document (GitHub Pages, opening from disk).
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<meta name="description" content="Himalayan radar nowcast, pressure, snow line, river flow and seasonal ecology.">'
            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + body + "</body></html>")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    # artifact.html: the bare fragment; the claude.ai artifact host adds its own skeleton.
    with open(os.path.join(out_dir, "artifact.html"), "w", encoding="utf-8") as f:
        f.write(body)
    log(f"wrote {args.out} ({len(html) / 1e6:.1f} MB): {len(r_times)} observed + {len(fc)} extrapolated frames, "
        f"{len(stations)} stations, {len(gauges)} gauges, {len(evs)} events")


if __name__ == "__main__":
    sys.exit(main())
