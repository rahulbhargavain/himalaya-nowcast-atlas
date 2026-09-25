"""Radar: RainViewer composite frames, optical-flow motion, and an 8-hour
Lagrangian-persistence extrapolation.

Method
  1. Decode the last ~2 h of RainViewer frames (10-min spacing) to dBZ
     using RainViewer's published colour table.
  2. Dense optical flow (OpenCV DIS) between consecutive frames, averaged
     over the recent pairs where echo exists. The flow is smoothed by
     normalised convolution so it describes storm-scale motion, not pixel
     noise.
  3. Away from echo (and outside radar coverage) the motion falls back to
     the model's mean 500/700 hPa steering wind, blended by distance from
     echo.
  4. Semi-Lagrangian backward trajectories carry the latest frame forward
     in 10-minute steps for 8 hours. Small-scale detail is blurred as lead
     time grows, since cells that small are not predictable that far ahead.

Extrapolation assumes storms keep their motion and intensity. Skill is
useful for roughly the first 1-2 hours and falls off after that; the page
says so and shows the model's rain for comparison.
"""
from __future__ import annotations

import csv
import io

import cv2
import numpy as np
from PIL import Image

from .geo import CACHE, FRAME, fetch_many, get, get_json, stitch_tiles

STEP_MIN = 10
LEAD_STEPS = 48          # 8 h
MIN_DBZ = 12.0


def _palette():
    body = get("https://www.rainviewer.com/files/rainviewer_api_colors_table.csv",
               cache_key="rv_colors.csv")
    rows = list(csv.reader(io.StringIO(body.decode("utf-8"))))
    col = rows[0].index("Universal Blue")
    keys, vals = [], []
    for r in rows[1:]:
        hexv = r[col].lstrip("#")
        if len(hexv) != 8 or hexv.endswith("00"):
            continue
        rgb = int(hexv[:6], 16)
        keys.append(rgb)
        vals.append(float(r[0]))
    keys = np.array(keys, np.int64)
    vals = np.array(vals, np.float32)
    rgb = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], -1).astype(np.float32)
    return keys, vals, rgb


def decode_dbz(im_rgba, pal):
    keys, vals, rgb = pal
    a = np.asarray(im_rgba)
    alpha = a[..., 3] > 0
    out = np.zeros(a.shape[:2], np.float32)
    if not alpha.any():
        return out
    px = a[alpha][:, :3].astype(np.float32)
    # Exact matches are the norm (smoothing off); nearest colour handles the rest.
    uniq, inv = np.unique(px, axis=0, return_inverse=True)
    d = ((uniq[:, None, :] - rgb[None, :, :]) ** 2).sum(-1)
    out[alpha] = vals[d.argmin(1)][inv.ravel()]
    return out


def fetch_frames():
    """Returns (times[], dbz stack (n, gh, gw), coverage mask (gh, gw), host)."""
    meta = get_json("https://api.rainviewer.com/public/weather-maps.json")
    host = meta["host"]
    past = meta["radar"]["past"]
    pal = _palette()
    # z6 tile pixels == GRID pixels (GRID is z7 halved).
    box = (FRAME.x0 / 2, FRAME.y0 / 2, FRAME.x0 / 2 + FRAME.gw, FRAME.y0 / 2 + FRAME.gh)

    def one(frame):
        url = host + frame["path"] + "/256/{z}/{x}/{y}/2/0_0.png"
        im = stitch_tiles(url, 6, box, cache_prefix="rv/" + frame["path"].strip("/").replace("/", "_"), workers=6)
        return decode_dbz(im, pal)

    stack = np.stack(fetch_many(one, past, workers=3))
    _prune_tile_cache({"rv/" + f["path"].strip("/").replace("/", "_") for f in past})
    cov = stitch_tiles(host + "/v2/coverage/0/256/{z}/{x}/{y}/0/0_0.png", 6, box, cache_prefix="rv/coverage")
    covered = np.asarray(cov)[..., 3] < 128  # the coverage layer paints NO-coverage areas
    return [f["time"] for f in past], stack, covered


def _prune_tile_cache(keep):
    """Drop cached tiles of radar frames no longer in the 2-hour window."""
    import os
    import shutil
    root = os.path.join(CACHE, "rv")
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        if name.startswith("v2_radar_") and "rv/" + name not in keep:
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _to_u8(dbz):
    return np.clip((dbz - 5.0) * (255.0 / 55.0), 0, 255).astype(np.uint8)


def motion_field(stack, steer_uv):
    """Per-pixel motion in GRID px per 10 min, shape (gh, gw, 2)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    n = stack.shape[0]
    acc = np.zeros(stack.shape[1:] + (2,), np.float32)
    wsum = np.zeros(stack.shape[1:], np.float32)
    for i in range(max(1, n - 7), n):
        a = cv2.GaussianBlur(_to_u8(stack[i - 1]), (0, 0), 1.5)
        b = cv2.GaussianBlur(_to_u8(stack[i]), (0, 0), 1.5)
        f = dis.calc(a, b, None)
        w = ((stack[i - 1] > MIN_DBZ) & (stack[i] > MIN_DBZ)).astype(np.float32)
        acc += f * w[..., None]
        wsum += w
    # Normalised convolution: smooth the flow where it is supported by echo.
    sig = 18.0
    num = np.stack([cv2.GaussianBlur(acc[..., k], (0, 0), sig) for k in range(2)], -1)
    den = cv2.GaussianBlur(wsum, (0, 0), sig)
    flow = num / np.maximum(den, 1e-6)[..., None]
    # Confidence ramps up with nearby echo support.
    conf = np.clip(den / 1.2, 0, 1)[..., None]
    v = conf * flow + (1 - conf) * steer_uv
    # Cap at ~150 km/h (GRID px is ~2.1 km at 30N) to reject bad matches.
    sp = np.linalg.norm(v, axis=-1, keepdims=True)
    cap = 150 / 6 / 2.1
    v = np.where(sp > cap, v * cap / np.maximum(sp, 1e-6), v)
    return v.astype(np.float32), conf[..., 0]


def extrapolate(dbz0, v, steps=LEAD_STEPS):
    gh, gw = dbz0.shape
    gx, gy = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
    px, py = gx.copy(), gy.copy()
    vx, vy = v[..., 0], v[..., 1]
    frames = []
    for k in range(1, steps + 1):
        # Backward trajectory: where did the air now at (x, y) come from?
        ux = cv2.remap(vx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        uy = cv2.remap(vy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        px -= ux
        py -= uy
        f = cv2.remap(dbz0, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sigma = 0.35 * k ** 0.65
        if sigma > 0.4:
            # Blur in linear rain-rate space so blurring doesn't invent echo.
            z = np.where(f > MIN_DBZ, 10 ** (f / 10.0), 0).astype(np.float32)
            z = cv2.GaussianBlur(z, (0, 0), sigma)
            f = np.where(z > 10 ** (MIN_DBZ / 10), 10 * np.log10(np.maximum(z, 1e-6)), 0).astype(np.float32)
        frames.append(f)
    return frames


# Radar colours: teal through ochre to crimson and violet. Kept apart from the
# model-rain palette (blues) so observed and modelled rain never look alike.
DBZ_STOPS = np.array([12, 20, 28, 36, 44, 52, 60], np.float32)
DBZ_RGBA = np.array([
    [120, 200, 190, 110], [48, 170, 160, 190], [150, 196, 70, 215], [236, 196, 58, 230],
    [232, 120, 40, 240], [205, 40, 60, 245], [150, 40, 170, 250],
], np.float32)


def colorize(dbz):
    rgba = np.stack([np.interp(dbz, DBZ_STOPS, DBZ_RGBA[:, i]) for i in range(4)], -1)
    rgba[dbz < MIN_DBZ] = 0
    return rgba.astype(np.uint8)


def rain_rate(dbz):
    """Marshall-Palmer Z = 200 R^1.6 -> mm/h."""
    return np.where(dbz > MIN_DBZ, (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6), 0.0)


def cells(dbz, v, lon_g, lat_g, places, threshold=35.0, min_px=6):
    """Strongest convective cells at T0 with heading and speed."""
    mask = (dbz >= threshold).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    km_per_px = 2.1
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        sel = lab == i
        cx, cy = cent[i]
        ix, iy = int(round(cx)), int(round(cy))
        mv = v[sel].mean(0)
        speed = float(np.hypot(*mv)) * km_per_px * 6
        heading = (np.degrees(np.arctan2(mv[0], -mv[1])) + 360) % 360
        lon, lat = float(lon_g[iy, ix]), float(lat_g[iy, ix])
        near = min(places, key=lambda p: (p["lat"] - lat) ** 2 + ((p["lon"] - lon) * np.cos(np.radians(lat))) ** 2)
        dist = 111 * np.hypot(near["lat"] - lat, (near["lon"] - lon) * np.cos(np.radians(lat)))
        out.append({
            "x": float(cx), "y": float(cy), "lon": round(lon, 2), "lat": round(lat, 2),
            "max_dbz": float(dbz[sel].max()), "area_km2": int(stats[i, cv2.CC_STAT_AREA] * km_per_px ** 2),
            "speed_kmh": round(speed), "heading": round(float(heading)),
            "near": near["name"] if dist <= 120 else None, "near_km": round(float(dist)),
        })
    out.sort(key=lambda c: (-c["max_dbz"], -c["area_km2"]))
    return out[:8]
