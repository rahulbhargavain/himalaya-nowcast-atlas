"""Model fields from Open-Meteo (best-match NWP) on a 0.75 degree grid:
MSL pressure (isobars, high/low centres), precipitation, freezing level
(-> snow line), and the 500/700 hPa steering wind used by the radar
extrapolation where there is no echo to track.

Snow line: the level where falling snow turns to rain sits on average
about 300 m below the 0 C freezing level. That offset is a rule of thumb
and varies with humidity and precipitation rate.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geo import E_LON, FRAME, N_LAT, S_LAT, W_LON, get_json, sample_latlon_grid

STEP = 0.75
SNOWLINE_OFFSET_M = 300
PLATEAU_M = 2500  # MSL pressure is an extrapolation above this; not drawn
VARS = ["pressure_msl", "precipitation", "freezing_level_height", "snowfall",
        "wind_speed_500hPa", "wind_direction_500hPa", "wind_speed_700hPa", "wind_direction_700hPa"]


def fetch_grid(past_hours=2, forecast_hours=9):
    lats = np.arange(S_LAT, N_LAT + 1e-6, STEP)
    lons = np.arange(W_LON, E_LON + 1e-6, STEP)
    pts = [(la, lo) for la in lats for lo in lons]
    chunks = [pts[i:i + 100] for i in range(0, len(pts), 100)]
    results = []
    for ch in chunks:
        j = get_json("https://api.open-meteo.com/v1/forecast", params=dict(
            latitude=",".join(f"{p[0]:.2f}" for p in ch), longitude=",".join(f"{p[1]:.2f}" for p in ch),
            hourly=",".join(VARS), past_hours=past_hours, forecast_hours=forecast_hours,
            timeformat="unixtime", timezone="GMT"), timeout=120)
        results.extend(j if isinstance(j, list) else [j])
    times = results[0]["hourly"]["time"]
    nt = len(times)
    shape = (nt, len(lats), len(lons))
    data = {v: np.full(shape, np.nan, np.float32) for v in VARS}
    elev = np.zeros(shape[1:], np.float32)
    for k, r in enumerate(results):
        i, j = divmod(k, len(lons))
        elev[i, j] = r.get("elevation") or 0
        for v in VARS:
            vals = r["hourly"].get(v)
            if vals:
                data[v][:, i, j] = [np.nan if x is None else x for x in vals]
    return {"times": times, "lats": lats, "lons": lons, "elev": elev, **data}


def steering_uv(g, t_index):
    """Mean of 500 and 700 hPa wind as GRID px per 10 min, (gh, gw, 2)."""
    uv = []
    for lvl in ("500hPa", "700hPa"):
        sp = g[f"wind_speed_{lvl}"][t_index] / 3.6            # m/s
        dr = np.radians(g[f"wind_direction_{lvl}"][t_index])  # from-direction
        u, v = -sp * np.sin(dr), -sp * np.cos(dr)             # to-east, to-north
        uv.append((np.nan_to_num(u), np.nan_to_num(v)))
    u = (uv[0][0] + uv[1][0]) / 2
    v = (uv[0][1] + uv[1][1]) / 2
    ug = sample_latlon_grid(u, g["lats"], g["lons"])
    vg = sample_latlon_grid(v, g["lats"], g["lons"])
    m_per_px = 2100.0
    return np.stack([ug * 600 / m_per_px, -vg * 600 / m_per_px], -1).astype(np.float32)


def _contour_paths(field, levels, scale=1.0, min_len=6, decimals=1):
    """Contours of a 2-D array as SVG path strings (one per level), in
    GRID pixel units (field sampled every `scale` GRID px)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    cs = plt.contour(field, levels=levels)
    out = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        parts = []
        for s in segs:
            if len(s) < min_len:
                continue
            s = s[:: 2] if len(s) > 60 else s
            parts.append("M" + "L".join(f"{x * scale:.{decimals}f},{y * scale:.{decimals}f}" for x, y in s))
        if parts:
            out.append({"level": float(lev), "d": " ".join(parts)})
    plt.close(fig)
    return out


def pressure_layers(g, dem_g):
    """Per hour: isobars (2 hPa) and H/L centres, masked over high terrain."""
    lats, lons = g["lats"], g["lons"]
    mask_hi = sample_latlon_grid(g["elev"], lats, lons) > PLATEAU_M
    hours = []
    all_p = g["pressure_msl"]
    lo_lev = np.floor(np.nanmin(all_p) / 2) * 2
    hi_lev = np.ceil(np.nanmax(all_p) / 2) * 2
    levels = np.arange(lo_lev, hi_lev + 0.1, 2)
    for t in range(len(g["times"])):
        p = np.nan_to_num(all_p[t], nan=np.nanmean(all_p[t]))
        p = cv2.GaussianBlur(p, (0, 0), 0.8)
        pg = sample_latlon_grid(p, lats, lons)
        pg_masked = pg.copy()
        pg_masked[mask_hi | (dem_g > PLATEAU_M)] = np.nan
        q = 4
        small = pg_masked[::q, ::q]
        iso = _contour_paths(np.ma.masked_invalid(small), levels, scale=q)
        # H/L: extrema of the masked low-res grid within a 2.5 deg window.
        pm = p.copy()
        pm[g["elev"] > PLATEAU_M] = np.nan
        centres = []
        k = 5
        for i in range(1, len(lats) - 1):      # skip the grid edge: an edge
            for j in range(1, len(lons) - 1):  # cell can't be a real extremum
                c = pm[i, j]
                if not np.isfinite(c):
                    continue
                win = pm[max(0, i - k // 2): i + k // 2 + 1, max(0, j - k // 2): j + k // 2 + 1]
                if np.isnan(win).sum() > win.size // 2:
                    continue
                if c >= np.nanmax(win) and c - np.nanmean(win) > 0.6:
                    kind = "H"
                elif c <= np.nanmin(win) and np.nanmean(win) - c > 0.6:
                    kind = "L"
                else:
                    continue
                x, y = FRAME.to_grid(lons[j], lats[i])
                centres.append({"kind": kind, "hpa": round(float(c), 1), "x": float(x), "y": float(y),
                                "lon": float(lons[j]), "lat": float(lats[i])})
        hours.append({"iso": iso, "centres": centres})
    return hours


def snowline_layers(g, dem_g):
    """Per hour: snow line altitude field (m) and its contour on the terrain."""
    lats, lons = g["lats"], g["lons"]
    q = 4
    dem_s = cv2.resize(dem_g, (dem_g.shape[1] // q, dem_g.shape[0] // q), interpolation=cv2.INTER_AREA)
    # Smooth the terrain a little: on the Tibetan plateau the ground sits near
    # the snow line for hundreds of km and a raw contour turns into confetti.
    dem_s = cv2.GaussianBlur(dem_s, (0, 0), 1.2)
    out = []
    for t in range(len(g["times"])):
        fl = g["freezing_level_height"][t]
        fl = np.where(np.isfinite(fl), fl, np.nanmean(fl))
        sl = sample_latlon_grid(np.maximum(fl - SNOWLINE_OFFSET_M, 0), lats, lons)
        sl_s = cv2.resize(sl, (dem_s.shape[1], dem_s.shape[0]), interpolation=cv2.INTER_AREA)
        diff = np.where(dem_s > 1500, dem_s - sl_s, -9999)
        paths = _contour_paths(diff, [0.0], scale=q, min_len=16)
        out.append({"d": paths[0]["d"] if paths else "", "mean_m": int(np.nanmean(sl_s[dem_s > 3000])) if (dem_s > 3000).any() else None})
    return out


def point_series(g, lon, lat):
    """Bilinear sample of every hourly variable at one place."""
    lats, lons = g["lats"], g["lons"]
    fi = (lat - lats[0]) / STEP
    fj = (lon - lons[0]) / STEP
    i0, j0 = int(np.floor(fi)), int(np.floor(fj))
    di, dj = fi - i0, fj - j0
    i1, j1 = min(i0 + 1, len(lats) - 1), min(j0 + 1, len(lons) - 1)

    def s(arr):
        return (arr[:, i0, j0] * (1 - di) * (1 - dj) + arr[:, i1, j0] * di * (1 - dj)
                + arr[:, i0, j1] * (1 - di) * dj + arr[:, i1, j1] * di * dj)
    return {v: s(g[v]) for v in ("precipitation", "freezing_level_height", "snowfall", "pressure_msl")}
