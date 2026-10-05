"""Regional nowcast pages (Kumaon, Garhwal): standalone pages for one
division of Uttarakhand each, built in the same run as the Himalaya atlas.

Each reuses the atlas's radar extrapolation (cropped, so storms moving in
from outside are already tracked) and model grid, and adds what only makes
sense at this scale: zoom-10 relief, IMD's own Doppler radars in the hills,
IMD's live automatic stations, nowcasts and 5-day district warnings, town
forecasts and air quality, and GloFAS flow on the region's rivers.

A region is a `Region` (REGIONS at the end of this file); build.py builds
every one of them into dist/<id>/.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import icimod, imd_nowcast, imd_obs, land, model, radar, satimg, satrain, water
from .geo import CACHE, FRAME, get_json, make_frame, world_lonlat

HOURLY = ["temperature_2m", "precipitation", "precipitation_probability", "weather_code",
          "cloud_cover", "wind_gusts_10m", "freezing_level_height", "cape"]
AQ_HOURLY = ["pm2_5", "pm10", "us_aqi"]


@dataclass
class Region:
    id: str                 # "kumaon": the page is dist/<id>/index.html
    name: str               # "Kumaon"
    box: tuple              # w, s, e, n
    districts: list         # IMD's spellings (upper case), in display order
    towns: list             # [(name, lat, lon, district)]; forecast points
    gauges: list            # [(river, place, lat, lon)]; GloFAS points
    radars: list            # IMD radar codes (imd_obs.RADARS) covering the region
    home: str               # the town selected by default
    store: str              # localStorage key prefix (the pages share an origin)
    shade_alt: str
    station_note: str = ""  # appended to the observed-stations note
    gauge_reach: float = 0.05
    z: int = 10

    @property
    def frame(self):
        w, s, e, n = self.box
        return make_frame(w, e, s, n, z=self.z)

    def in_box(self, lon, lat, pad=0.0):
        w, s, e, n = self.box
        return w - pad <= lon <= e + pad and s - pad <= lat <= n + pad

    @property
    def last_path(self):
        # Kumaon's predates regions: cache/kumaon_last.json either way.
        return os.path.join(CACHE, f"{self.id}_last.json")


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


# --------------------------------------------------------------------------
# Rasters
# --------------------------------------------------------------------------

def _terrain(frame):
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


def _radar_crop(R, F, frames_all, covered):
    """Crop the atlas's radar frames (zoom-6 pixels) to the region, and say
    where the crop sits on this page's grid."""
    xh0, yh0 = FRAME.to_grid(R.box[0], R.box[3])
    xh1, yh1 = FRAME.to_grid(R.box[2], R.box[1])
    ix0, iy0 = int(np.floor(xh0)) - 1, int(np.floor(yh0)) - 1
    ix1, iy1 = int(np.ceil(xh1)) + 1, int(np.ceil(yh1)) + 1
    scale = 2 ** (F.z - FRAME.z)
    gx0 = ((FRAME.x0 + 2 * ix0) * scale - F.x0) / 2
    gy0 = ((FRAME.y0 + 2 * iy0) * scale - F.y0) / 2
    cov = covered[iy0:iy1, ix0:ix1]
    place = {"x": gx0, "y": gy0, "w": (ix1 - ix0) * scale, "h": (iy1 - iy0) * scale}
    crops = [f[iy0:iy1, ix0:ix1] for f in frames_all]
    return crops, place, float(cov.mean())


def _imd_radars(R, F, attempt):
    """Each of the region's IMD radars on the page grid, and their maximum
    as one layer (where two overlap, the stronger echo shows)."""
    out, comp = [], None
    for code in R.radars:
        r = attempt(f"{imd_obs.RADARS[code]['name']} radar", lambda c=code: imd_obs.radar(c, F), None)
        if r is None:
            continue
        d = r.pop("dbz", None)
        if d is not None:
            comp = d if comp is None else np.maximum(comp, d)
        out.append(r)
    ts = [r["t"] for r in out if r.get("t")]
    layer = {"src": _png(radar.colorize(comp), quantize=48) if comp is not None else None,
             "t": max(ts) if ts else None, "echo_px": int((comp > 0).sum()) if comp is not None else None}
    return out, layer, comp


# --------------------------------------------------------------------------
# Points
# --------------------------------------------------------------------------

def _forecasts(R):
    lats = ",".join(f"{t[1]:.3f}" for t in R.towns)
    lons = ",".join(f"{t[2]:.3f}" for t in R.towns)
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
        aq = [None] * len(R.towns)
    return fc, aq


def _clean(vals, nd=1):
    return [None if v is None else round(float(v), nd) for v in vals]


def _towns(R, F, fc, aq, frames_all, t_all, t0_index, t0, imd_dbz):
    out = []
    for k, (name, lat, lon, district) in enumerate(R.towns):
        x, y = F.to_grid(lon, lat)
        hx, hy = FRAME.to_grid(lon, lat)
        ix, iy = int(round(float(hx))), int(round(float(hy)))
        dbz = [float(f[iy - 3:iy + 4, ix - 3:ix + 4].max()) for f in frames_all]
        onset = next((int((t_all[j] - t0) / 60) for j in range(t0_index + 1, len(dbz)) if dbz[j] >= 25), None)
        imd_now = None
        if imd_dbz is not None:
            r = int(round(3 / F.km_per_grid_px))
            kx, ky = int(round(float(x))), int(round(float(y)))
            win = imd_dbz[max(0, ky - r):ky + r + 1, max(0, kx - r):kx + r + 1]
            imd_now = float(win.max()) if win.size else None
        h = fc[k]["hourly"]
        rec = {"name": name, "district": district, "lat": lat, "lon": lon, "x": float(x), "y": float(y),
               "elev": fc[k].get("elevation"),
               "dbz_now": round(dbz[t0_index]), "radar_onset_min": onset if dbz[t0_index] < 25 else 0,
               "mks_dbz": imd_now,
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


def _model_subset(R, g):
    lats, lons = g["lats"], g["lons"]
    step = model.STEP
    i = np.where((lats >= R.box[1] - step) & (lats <= R.box[3] + step))[0]
    j = np.where((lons >= R.box[0] - step) & (lons <= R.box[2] + step))[0]
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


def _fire_clusters(R, F, fires):
    """Fire pixel clusters with their centre, from the overlay's alpha."""
    if not fires.get("src"):
        return []
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(fires["src"].split(",", 1)[1]))).convert("RGBA"))
    n, lab, st, cen = cv2.connectedComponentsWithStats((a[..., 3] > 0).astype(np.uint8), connectivity=8)
    lon_g, lat_g = F.grid_lonlat()
    out = []
    for i in range(1, n):
        cx, cy = cen[i]
        lon, lat = float(lon_g[int(cy), int(cx)]), float(lat_g[int(cy), int(cx)])
        near = min(R.towns, key=lambda t: (t[1] - lat) ** 2 + ((t[2] - lon) * 0.87) ** 2)
        km = 111 * np.hypot(near[1] - lat, (near[2] - lon) * 0.87)
        out.append({"x": float(cx), "y": float(cy), "lat": round(lat, 3), "lon": round(lon, 3),
                    "px": int(st[i, cv2.CC_STAT_AREA]), "near": near[0], "near_km": round(float(km))})
    return sorted(out, key=lambda c: -c["px"])


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(HERE, "regional_template.html")


def _page_meta(R):
    """What the template needs to know about the region, and its links."""
    names = [d.title() for d in R.districts]
    return {"id": R.id, "name": R.name, "imd_districts": R.districts, "districts": names,
            "district_list": ", ".join(names[:-1]) + " and " + names[-1], "home": R.home, "store": R.store,
            "shade_alt": R.shade_alt, "station_note": R.station_note,
            "radars": [imd_obs.RADARS[c]["name"] for c in R.radars],
            "others": [{"href": f"../{o.id}/", "name": o.name} for o in REGIONS.values() if o.id != R.id]}


def render(data, out_path, template_path=TEMPLATE):
    R = REGIONS[data.get("region", {}).get("id", "kumaon")]
    data["region"] = _page_meta(R)  # links and wording always current, even on a re-render
    tpl = open(template_path, encoding="utf-8").read()
    body = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    body = body.replace("<title>Regional Nowcast</title>", f"<title>{R.name} Nowcast</title>")
    desc = (f"{R.name} nowcast: radar, IMD warnings and stations, town forecasts, rivers, fires and snow.")
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            f'<meta name="description" content="{desc}">'
            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + body + "</body></html>")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return len(html)


def build(R, ctx, out_path, template_path=TEMPLATE, log=print):
    """ctx: the atlas build's g, frames_all, t_all, t0_index, t0, covered,
    quakes, step_min. Writes the page and returns a one-line summary."""
    F = R.frame
    g, t0 = ctx["g"], ctx["t0"]
    notes = []

    def attempt(label, fn, default):
        try:
            return fn()
        except Exception as e:  # one source being down shouldn't sink the page
            log(f"  {R.id} {label} failed: {e}")
            notes.append(label)
            return default

    log(f"{R.id}: terrain")
    dem_g, shade = _terrain(F)

    log(f"{R.id}: radar crop, IMD radars")
    crops, place, cov_frac = _radar_crop(R, F, ctx["frames_all"], ctx["covered"])
    radar_frames = [{"t": int(t), "obs": k <= ctx["t0_index"], "src": _png(radar.colorize(f), quantize=48)}
                    for k, (t, f) in enumerate(zip(ctx["t_all"], crops))]
    radars, imd_layer, imd_dbz = _imd_radars(R, F, attempt)

    log(f"{R.id}: satellite (IMERG, true colour, fires, snow)")
    imerg = attempt("IMERG", lambda: satrain.latest(frame=F), {"src": None, "t": None})
    truecolor = attempt("true colour", lambda: satimg.true_color(frame=F), {"src": None})
    fires = attempt("fires", lambda: icimod.live_fires(2, frame=F), {"src": None, "pixels": 0, "dates": []})
    fires["clusters"] = _fire_clusters(R, F, fires)
    snow, snow_dates = attempt("snow", lambda: land.recent_snow(8, frame=F), (np.zeros((F.gh, F.gw)), (None, None)))
    snow_km2 = int((snow >= 40).sum() * F.km_per_grid_px ** 2)

    log(f"{R.id}: IMD warnings, nowcasts and stations")
    box = R.box
    imd = {
        "districts": attempt("IMD district nowcast", lambda: imd_nowcast.districts(F, box, eps=0.003), []),
        "stations": attempt("IMD station nowcast", lambda: imd_nowcast.stations(F, box), []),
        "warnings": attempt("IMD district warnings", lambda: imd_nowcast.district_warnings(R.districts), []),
        "synop": attempt("IMD SYNOP", lambda: imd_obs.synop(box, F), []),
    }
    imd["aws"], imd["aws_silent"] = attempt("IMD AWS", lambda: imd_obs.aws(box, F), ([], 0))

    log(f"{R.id}: town forecasts and air quality")
    fc, aq = _forecasts(R)
    towns = _towns(R, F, fc, aq, ctx["frames_all"], ctx["t_all"], ctx["t0_index"], t0, imd_dbz)

    log(f"{R.id}: rivers")
    gauges = attempt("gauges", lambda: water.gauges(R.gauges, tag=f"_{R.id}", reach=R.gauge_reach, frame=F), [])

    quakes = []
    for q in ctx.get("quakes") or []:
        if R.in_box(q["lon"], q["lat"], pad=1.0):
            x, y = F.to_grid(q["lon"], q["lat"])
            quakes.append({**q, "x": float(x), "y": float(y), "inside": R.in_box(q["lon"], q["lat"])})

    snowline = model.snowline_layers(g, dem_g, frame=F, q=2)

    data = {
        "region": {"id": R.id},
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()), "t0": int(t0), "step_min": ctx["step_min"],
        "frame": {"w": F.w, "h": F.h, "gw": F.gw, "gh": F.gh, "x0": F.x0, "y0": F.y0, "z": F.z,
                  "box": list(R.box), "km_px": F.km_per_grid_px},
        "images": {"shade": _jpg(shade), "dem": _dem_png(dem_g)},
        "radar": {"frames": radar_frames, "t0_index": ctx["t0_index"], "place": place, "coverage": cov_frac,
                  "dbz_stops": radar.DBZ_STOPS.tolist(), "dbz_rgba": radar.DBZ_RGBA.tolist()},
        "mks": imd_layer, "radars": radars, "imerg": imerg, "truecolor": truecolor,
        "fires": fires, "snow": {"src": _snow_png(snow), "dates": [str(d) if d else None for d in snow_dates],
                                 "km2": snow_km2},
        "model": _model_subset(R, g), "snowline": snowline,
        "towns": towns, "events": _events(towns, t0), "imd": imd, "gauges": gauges, "quakes": quakes,
        "failed": notes,
    }
    with open(R.last_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    size = render(data, out_path, template_path)
    return (f"{size / 1e6:.1f} MB, {len(towns)} towns, {len(imd['aws'])} live IMD stations, "
            f"IMD radar echo {imd_layer.get('echo_px') or 0} px from {len(radars)} radars, "
            f"RainViewer coverage {cov_frac:.0%}, {len(fires['clusters'])} fire clusters, {len(gauges)} gauges"
            + (f", failed: {', '.join(notes)}" if notes else ""))


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------

KUMAON = Region(
    id="kumaon", name="Kumaon", box=(78.6, 28.6, 81.2, 30.7),
    districts=["NAINITAL", "ALMORA", "BAGESHWAR", "CHAMPAWAT", "PITHORAGARH", "UDHAM SINGH NAGAR"],
    towns=[
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
    ],
    gauges=[
        ("Kosi", "Ramnagar", 29.395, 79.135), ("Gaula", "Kathgodam", 29.268, 79.540),
        ("Ramganga (West)", "Kalagarh", 29.530, 78.760), ("Saryu", "Bageshwar", 29.838, 79.773),
        ("Ramganga (East)", "Thal", 29.833, 80.143), ("Kali (with Gori)", "Jauljibi", 29.753, 80.383),
        ("Kali", "Dharchula", 29.847, 80.540), ("Sharda", "Tanakpur", 29.070, 80.110),
    ],
    radars=["mks"], home="Bhimtal", store="kna:",
    shade_alt="Shaded relief of Kumaon, from the Terai plains to the Panchachuli and Nanda Devi massifs",
    station_note="There is no public IMD station at Bhimtal itself.")

GARHWAL = Region(
    id="garhwal", name="Garhwal", box=(77.5, 29.35, 80.2, 31.55),
    districts=["UTTARKASHI", "CHAMOLI", "RUDRAPRAYAG", "TEHRI GARHWAL", "PAURI GARHWAL", "DEHRADUN", "HARIDWAR"],
    # At their OSM place nodes (shared with the Garhwal cultural atlas).
    towns=[
        ("Dehradun", 30.326, 78.044, "Dehradun"), ("Mussoorie", 30.457, 78.078, "Dehradun"),
        ("Vikasnagar", 30.469, 77.771, "Dehradun"), ("Chakrata", 30.701, 77.870, "Dehradun"),
        ("Rishikesh", 30.109, 78.292, "Dehradun"), ("Haridwar", 29.938, 78.145, "Haridwar"),
        ("Roorkee", 29.869, 77.890, "Haridwar"), ("Kotdwar", 29.746, 78.520, "Pauri Garhwal"),
        ("Lansdowne", 29.838, 78.682, "Pauri Garhwal"), ("Pauri", 30.148, 78.776, "Pauri Garhwal"),
        ("Srinagar", 30.223, 78.790, "Pauri Garhwal"), ("Devprayag", 30.146, 78.599, "Tehri Garhwal"),
        ("New Tehri", 30.374, 78.433, "Tehri Garhwal"), ("Chamba", 30.346, 78.394, "Tehri Garhwal"),
        ("Dhanaulti", 30.425, 78.246, "Tehri Garhwal"), ("Uttarkashi", 30.729, 78.440, "Uttarkashi"),
        ("Barkot", 30.809, 78.206, "Uttarkashi"), ("Purola", 30.879, 78.083, "Uttarkashi"),
        ("Harsil", 31.037, 78.751, "Uttarkashi"), ("Gangotri", 30.994, 78.940, "Uttarkashi"),
        ("Rudraprayag", 30.287, 78.981, "Rudraprayag"), ("Ukhimath", 30.514, 79.095, "Rudraprayag"),
        ("Gaurikund", 30.653, 79.027, "Rudraprayag"), ("Gopeshwar", 30.409, 79.319, "Chamoli"),
        ("Karnaprayag", 30.260, 79.217, "Chamoli"), ("Gauchar", 30.289, 79.157, "Chamoli"),
        ("Joshimath", 30.556, 79.561, "Chamoli"), ("Badrinath", 30.742, 79.493, "Chamoli"),
    ],
    # Points a little away from the confluences, so the snap to the largest
    # GloFAS channel nearby finds the named river.
    gauges=[
        ("Yamuna", "Barkot", 30.810, 78.210), ("Tons", "Tiuni", 30.945, 77.855),
        ("Bhagirathi", "Uttarkashi", 30.730, 78.445), ("Alaknanda", "Joshimath", 30.548, 79.575),
        ("Pindar", "Tharali", 30.075, 79.520), ("Alaknanda (with Mandakini)", "Rudraprayag", 30.287, 78.985),
        ("Alaknanda", "Srinagar", 30.220, 78.780), ("Ganga", "Rishikesh", 30.105, 78.300),
        ("Ganga", "Haridwar", 29.950, 78.165),
    ],
    radars=["sur", "ldn", "mks"], home="Dehradun", store="gna:",
    shade_alt="Shaded relief of Garhwal, from the Doon valley and the Haridwar plains to Bandarpunch, "
              "the Gangotri peaks and Nanda Devi")

REGIONS = {R.id: R for R in (KUMAON, GARHWAL)}

if __name__ == "__main__":
    # Re-render the pages from the last build's data (template work).
    for R in REGIONS.values():
        if os.path.exists(R.last_path):
            data = json.load(open(R.last_path, encoding="utf-8"))
            data.setdefault("region", {"id": R.id})
            if "radars" not in data:  # a Kumaon build from before regions
                m = data.get("mks") or {}
                data["radars"] = [{"code": "mks", "name": "Mukteshwar", **{k: m.get(k) for k in (
                    "t", "image", "site", "range_km", "x", "y", "r_px", "echo_px", "note")}}]
            out = os.path.join(HERE, "dist", R.id, "index.html")
            print(f"wrote dist/{R.id}/index.html ({render(data, out) / 1e6:.1f} MB)")
