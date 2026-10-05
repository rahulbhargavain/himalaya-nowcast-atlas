"""Kumaon nowcast: a standalone page for Uttarakhand's Kumaon division
(Nainital, Almora, Bageshwar, Champawat, Pithoragarh and Udham Singh Nagar),
built in the same run as the Himalaya atlas.

It reuses the atlas's radar extrapolation (cropped, so storms moving in
from outside Kumaon are already tracked) and model grid, and adds what only
makes sense at this scale: zoom-10 relief, IMD's Mukteshwar Doppler radar,
IMD's live automatic stations, nowcasts and 5-day district warnings, town
forecasts and air quality, and GloFAS flow on Kumaon's own rivers.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os

import cv2
import numpy as np
from PIL import Image

from . import icimod, imd_nowcast, imd_obs, land, model, radar, satimg, satrain, water
from .geo import CACHE, FRAME, get_json, make_frame, world_lonlat

BOX = (78.6, 28.6, 81.2, 30.7)  # w, s, e, n
KF = make_frame(BOX[0], BOX[2], BOX[1], BOX[3], z=10)
DISTRICTS = ["NAINITAL", "ALMORA", "BAGESHWAR", "CHAMPAWAT", "PITHORAGARH", "UDHAM SINGH NAGAR"]

TOWNS = [
    ("Bhimtal", 29.345, 79.560, "Nainital"), ("Nainital", 29.380, 79.464, "Nainital"),
    ("Bhowali", 29.386, 79.520, "Nainital"), ("Mukteshwar", 29.472, 79.646, "Nainital"),
    ("Haldwani", 29.218, 79.513, "Nainital"), ("Ramnagar", 29.395, 79.127, "Nainital"),
    ("Almora", 29.597, 79.659, "Almora"), ("Ranikhet", 29.643, 79.432, "Almora"),
    ("Dwarahat", 29.779, 79.427, "Almora"), ("Kausani", 29.845, 79.603, "Bageshwar"),
    ("Bageshwar", 29.838, 79.771, "Bageshwar"), ("Munsiyari", 30.068, 80.239, "Pithoragarh"),
    ("Dharchula", 29.847, 80.537, "Pithoragarh"), ("Pithoragarh", 29.583, 80.218, "Pithoragarh"),
    ("Berinag", 29.775, 80.058, "Pithoragarh"), ("Champawat", 29.337, 80.091, "Champawat"),
    ("Lohaghat", 29.405, 80.087, "Champawat"), ("Tanakpur", 29.074, 80.111, "Champawat"),
    ("Khatima", 28.921, 79.970, "Udham Singh Nagar"), ("Rudrapur", 28.975, 79.400, "Udham Singh Nagar"),
    ("Kashipur", 29.210, 78.962, "Udham Singh Nagar"), ("Pantnagar", 29.030, 79.470, "Udham Singh Nagar"),
]

GAUGES = [
    ("Kosi", "Ramnagar", 29.395, 79.135), ("Gaula", "Kathgodam", 29.268, 79.540),
    ("Ramganga (West)", "Kalagarh", 29.530, 78.760), ("Saryu", "Bageshwar", 29.838, 79.773),
    ("Ramganga (East)", "Thal", 29.833, 80.143), ("Kali (with Gori)", "Jauljibi", 29.753, 80.383),
    ("Kali", "Dharchula", 29.847, 80.540), ("Sharda", "Tanakpur", 29.070, 80.110),
]

HOURLY = ["temperature_2m", "precipitation", "precipitation_probability", "weather_code",
          "cloud_cover", "wind_gusts_10m", "freezing_level_height", "cape"]
AQ_HOURLY = ["pm2_5", "pm10", "us_aqi"]


def _png(arr, mode="RGBA", quantize=None):
    im = Image.fromarray(arr, mode)
    if quantize:
        im = im.quantize(colors=quantize, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpg(arr, q=80):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _in_box(lon, lat, pad=0.0):
    w, s, e, n = BOX
    return w - pad <= lon <= e + pad and s - pad <= lat <= n + pad


# --------------------------------------------------------------------------
# Rasters
# --------------------------------------------------------------------------

def _terrain(frame=KF):
    """Elevation on the frame's GRID and the shaded relief at FULL
    resolution (the cultural atlases pass their own frame)."""
    dem = land.load_dem(frame)
    lat_rows = world_lonlat(np.full(frame.h, frame.x0), frame.y0 + np.arange(frame.h) + 0.5, frame.z)[1]
    shade = land.hillshade_image(dem, lat_rows, z=frame.z)
    dem_g = cv2.resize(dem, (frame.gw, frame.gh), interpolation=cv2.INTER_AREA)
    return dem_g, shade


def _dem_png(dem_g):
    """Elevation for the hover readout, at half the grid resolution (~530 m):
    plenty for a readout, a third of the bytes."""
    half = cv2.resize(dem_g, (dem_g.shape[1] // 2, dem_g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    u = np.clip(np.round(half) + 500, 0, 65535).astype(np.uint16)
    rgb = np.dstack([(u >> 8).astype(np.uint8), (u & 255).astype(np.uint8), np.zeros_like(u, np.uint8)])
    return _png(rgb, "RGB")


def _snow_png(snow):
    a = np.zeros(snow.shape + (4,), np.uint8)
    m = snow >= 10
    a[m] = [236, 246, 255, 0]
    a[..., 3][m] = np.minimum(235, 60 + snow[m] * 1.8).astype(np.uint8)
    return _png(a, quantize=32) if m.any() else None


def _radar_crop(frames_all, covered):
    """Crop the atlas's radar frames (zoom-6 pixels) to Kumaon, and say where
    the crop sits on this page's grid."""
    xh0, yh0 = FRAME.to_grid(BOX[0], BOX[3])
    xh1, yh1 = FRAME.to_grid(BOX[2], BOX[1])
    ix0, iy0 = int(np.floor(xh0)) - 1, int(np.floor(yh0)) - 1
    ix1, iy1 = int(np.ceil(xh1)) + 1, int(np.ceil(yh1)) + 1
    scale = 2 ** (KF.z - FRAME.z)
    gx0 = ((FRAME.x0 + 2 * ix0) * scale - KF.x0) / 2
    gy0 = ((FRAME.y0 + 2 * iy0) * scale - KF.y0) / 2
    cov = covered[iy0:iy1, ix0:ix1]
    place = {"x": gx0, "y": gy0, "w": (ix1 - ix0) * scale, "h": (iy1 - iy0) * scale}
    crops = [f[iy0:iy1, ix0:ix1] for f in frames_all]
    return crops, place, float(cov.mean())


# --------------------------------------------------------------------------
# Points
# --------------------------------------------------------------------------

def _forecasts():
    lats = ",".join(f"{t[1]:.3f}" for t in TOWNS)
    lons = ",".join(f"{t[2]:.3f}" for t in TOWNS)
    fc = get_json("https://api.open-meteo.com/v1/forecast", params=dict(
        latitude=lats, longitude=lons, hourly=",".join(HOURLY), daily="sunrise,sunset",
        forecast_hours=72, past_hours=1, timeformat="unixtime", timezone="Asia/Kolkata"), timeout=120)
    fc = fc if isinstance(fc, list) else [fc]
    try:
        aq = get_json("https://air-quality-api.open-meteo.com/v1/air-quality", params=dict(
            latitude=lats, longitude=lons, hourly=",".join(AQ_HOURLY), forecast_hours=48, past_hours=1,
            timeformat="unixtime", timezone="Asia/Kolkata"), timeout=120)
        aq = aq if isinstance(aq, list) else [aq]
    except Exception:  # air quality is a nice-to-have
        aq = [None] * len(TOWNS)
    return fc, aq


def _clean(vals, nd=1):
    return [None if v is None else round(float(v), nd) for v in vals]


def _towns(fc, aq, frames_all, t_all, t0_index, t0, mks_dbz):
    out = []
    for k, (name, lat, lon, district) in enumerate(TOWNS):
        x, y = KF.to_grid(lon, lat)
        hx, hy = FRAME.to_grid(lon, lat)
        ix, iy = int(round(float(hx))), int(round(float(hy)))
        dbz = [float(f[iy - 3:iy + 4, ix - 3:ix + 4].max()) for f in frames_all]
        onset = next((int((t_all[j] - t0) / 60) for j in range(t0_index + 1, len(dbz)) if dbz[j] >= 25), None)
        mks_now = None
        if mks_dbz is not None:
            r = int(round(3 / KF.km_per_grid_px))
            kx, ky = int(round(float(x))), int(round(float(y)))
            win = mks_dbz[max(0, ky - r):ky + r + 1, max(0, kx - r):kx + r + 1]
            mks_now = float(win.max()) if win.size else None
        h = fc[k]["hourly"]
        rec = {"name": name, "district": district, "lat": lat, "lon": lon, "x": float(x), "y": float(y),
               "elev": fc[k].get("elevation"),
               "dbz_now": round(dbz[t0_index]), "radar_onset_min": onset if dbz[t0_index] < 25 else 0,
               "mks_dbz": mks_now,
               "fc": {"t": h["time"], "temp": _clean(h["temperature_2m"]), "rain": _clean(h["precipitation"]),
                      "pop": _clean(h["precipitation_probability"], 0), "code": _clean(h["weather_code"], 0),
                      "cloud": _clean(h["cloud_cover"], 0), "gust": _clean(h["wind_gusts_10m"], 0),
                      "freeze": _clean(h["freezing_level_height"], 0), "cape": _clean(h["cape"], 0)},
               "sun": {"rise": fc[k]["daily"]["sunrise"], "set": fc[k]["daily"]["sunset"]}}
        a = aq[k] if k < len(aq) else None
        if a and a.get("hourly"):
            ah = a["hourly"]
            rec["aq"] = {"t": ah["time"], "pm25": _clean(ah["pm2_5"]), "pm10": _clean(ah["pm10"]),
                         "aqi": _clean(ah["us_aqi"], 0)}
        out.append(rec)
    return out


def _model_subset(g):
    lats, lons = g["lats"], g["lons"]
    step = model.STEP
    i = np.where((lats >= BOX[1] - step) & (lats <= BOX[3] + step))[0]
    j = np.where((lons >= BOX[0] - step) & (lons <= BOX[2] + step))[0]
    sub = np.ix_(range(len(g["times"])), i, j)
    precip = np.nan_to_num(g["precipitation"][sub], nan=0)
    freeze = np.nan_to_num(g["freezing_level_height"][sub], nan=0)
    return {"times": [int(t) for t in g["times"]], "lats": lats[i].tolist(), "lons": lons[j].tolist(),
            "precip10": np.round(precip * 10).astype(int).tolist(),
            "freeze10": np.round(freeze / 10).astype(int).tolist(), "snow_offset": model.SNOWLINE_OFFSET_M}


def _events(towns, t0):
    ev = []
    on = {}
    for tw in towns:
        if tw["radar_onset_min"]:
            on.setdefault(tw["radar_onset_min"], []).append(tw["name"])
    for t, names in on.items():
        ev.append({"t": t, "kind": "radar", "text": "Extrapolated echo reaches " + ", ".join(names)})
    rain_on = {}
    for tw in towns:
        for t, r in zip(tw["fc"]["t"], tw["fc"]["rain"]):
            if t > t0 and r is not None and r >= 1.0:
                rain_on.setdefault(int((t - t0) / 60), []).append(f"{tw['name']} {r:g}")
                break
    for t, items in rain_on.items():
        ev.append({"t": t, "kind": "model", "text": "Forecast rain from " + ", ".join(items) + " mm/h"})
    ev.sort(key=lambda e: e["t"])
    return [e for e in ev if 0 < e["t"] <= 480][:30]


def _fire_clusters(fires):
    """Fire pixel clusters with their centre, from the overlay's alpha."""
    if not fires.get("src"):
        return []
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(fires["src"].split(",", 1)[1]))).convert("RGBA"))
    n, lab, st, cen = cv2.connectedComponentsWithStats((a[..., 3] > 0).astype(np.uint8), connectivity=8)
    lon_g, lat_g = KF.grid_lonlat()
    out = []
    for i in range(1, n):
        cx, cy = cen[i]
        lon, lat = float(lon_g[int(cy), int(cx)]), float(lat_g[int(cy), int(cx)])
        near = min(TOWNS, key=lambda t: (t[1] - lat) ** 2 + ((t[2] - lon) * 0.87) ** 2)
        km = 111 * np.hypot(near[1] - lat, (near[2] - lon) * 0.87)
        out.append({"x": float(cx), "y": float(cy), "lat": round(lat, 3), "lon": round(lon, 3),
                    "px": int(st[i, cv2.CC_STAT_AREA]), "near": near[0], "near_km": round(float(km))})
    return sorted(out, key=lambda c: -c["px"])


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

LAST = os.path.join(CACHE, "kumaon_last.json")


def render(data, out_path, template_path):
    tpl = open(template_path, encoding="utf-8").read()
    body = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<meta name="description" content="Kumaon nowcast: radar, IMD warnings and stations, town forecasts, rivers, fires and snow.">'
            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + body + "</body></html>")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return len(html)


def build(ctx, out_path, template_path, log=print):
    """ctx: the atlas build's g, frames_all, t_all, t0_index, t0, covered,
    quakes, step_min. Writes the page and returns a one-line summary."""
    g, t0 = ctx["g"], ctx["t0"]
    notes = []

    def attempt(label, fn, default):
        try:
            return fn()
        except Exception as e:  # one source being down shouldn't sink the page
            log(f"  kumaon {label} failed: {e}")
            notes.append(label)
            return default

    log("kumaon: terrain")
    dem_g, shade = _terrain()

    log("kumaon: radar crop, Mukteshwar radar")
    crops, place, cov_frac = _radar_crop(ctx["frames_all"], ctx["covered"])
    radar_frames = [{"t": int(t), "obs": k <= ctx["t0_index"], "src": _png(radar.colorize(f), quantize=48)}
                    for k, (t, f) in enumerate(zip(ctx["t_all"], crops))]
    mks = attempt("Mukteshwar radar", lambda: imd_obs.mukteshwar(KF), {"t": None, "image": None, "dbz": None})
    mks_dbz = mks.pop("dbz", None)
    if mks_dbz is not None:
        mks["src"] = _png(radar.colorize(mks_dbz), quantize=48)

    log("kumaon: satellite (IMERG, true colour, fires, snow)")
    imerg = attempt("IMERG", lambda: satrain.latest(frame=KF), {"src": None, "t": None})
    truecolor = attempt("true colour", lambda: satimg.true_color(frame=KF), {"src": None})
    fires = attempt("fires", lambda: icimod.live_fires(2, frame=KF), {"src": None, "pixels": 0, "dates": []})
    fires["clusters"] = _fire_clusters(fires)
    snow, snow_dates = attempt("snow", lambda: land.recent_snow(8, frame=KF), (np.zeros((KF.gh, KF.gw)), (None, None)))
    snow_km2 = int((snow >= 40).sum() * KF.km_per_grid_px ** 2)

    log("kumaon: IMD warnings, nowcasts and stations")
    imd = {
        "districts": attempt("IMD district nowcast", lambda: imd_nowcast.districts(KF, BOX, eps=0.003), []),
        "stations": attempt("IMD station nowcast", lambda: imd_nowcast.stations(KF, BOX), []),
        "warnings": attempt("IMD district warnings", lambda: imd_nowcast.district_warnings(DISTRICTS), []),
        "synop": attempt("IMD SYNOP", lambda: imd_obs.synop(BOX, KF), []),
    }
    imd["aws"], imd["aws_silent"] = attempt("IMD AWS", lambda: imd_obs.aws(BOX, KF), ([], 0))

    log("kumaon: town forecasts and air quality")
    fc, aq = _forecasts()
    towns = _towns(fc, aq, ctx["frames_all"], ctx["t_all"], ctx["t0_index"], t0, mks_dbz)

    log("kumaon: rivers")
    gauges = attempt("gauges", lambda: water.gauges(GAUGES, tag="_kumaon", reach=0.05, frame=KF), [])

    quakes = []
    for q in ctx.get("quakes") or []:
        if _in_box(q["lon"], q["lat"], pad=1.0):
            x, y = KF.to_grid(q["lon"], q["lat"])
            quakes.append({**q, "x": float(x), "y": float(y), "inside": _in_box(q["lon"], q["lat"])})

    snowline = model.snowline_layers(g, dem_g, frame=KF, q=2)

    data = {
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()), "t0": int(t0), "step_min": ctx["step_min"],
        "frame": {"w": KF.w, "h": KF.h, "gw": KF.gw, "gh": KF.gh, "x0": KF.x0, "y0": KF.y0, "z": KF.z,
                  "box": list(BOX), "km_px": KF.km_per_grid_px},
        "images": {"shade": _jpg(shade), "dem": _dem_png(dem_g)},
        "radar": {"frames": radar_frames, "t0_index": ctx["t0_index"], "place": place, "coverage": cov_frac,
                  "dbz_stops": radar.DBZ_STOPS.tolist(), "dbz_rgba": radar.DBZ_RGBA.tolist()},
        "mks": mks, "imerg": imerg, "truecolor": truecolor,
        "fires": fires, "snow": {"src": _snow_png(snow), "dates": [str(d) if d else None for d in snow_dates],
                                 "km2": snow_km2},
        "model": _model_subset(g), "snowline": snowline,
        "towns": towns, "events": _events(towns, t0), "imd": imd, "gauges": gauges, "quakes": quakes,
        "failed": notes,
    }
    with open(LAST, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    size = render(data, out_path, template_path)
    return (f"{size / 1e6:.1f} MB, {len(towns)} towns, {len(imd['aws'])} live IMD stations, "
            f"Mukteshwar echo {mks.get('echo_px', 0)} px, RainViewer coverage {cov_frac:.0%}, "
            f"{len(fires['clusters'])} fire clusters, {len(gauges)} gauges" + (f", failed: {', '.join(notes)}" if notes else ""))


if __name__ == "__main__":
    # Re-render the page from the last build's data (template work).
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n = render(json.load(open(LAST, encoding="utf-8")), os.path.join(here, "dist", "kumaon", "index.html"),
               os.path.join(here, "kumaon_template.html"))
    print(f"wrote dist/kumaon/index.html ({n / 1e6:.1f} MB)")
