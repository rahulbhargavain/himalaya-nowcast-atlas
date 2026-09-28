"""NO2 air quality from Sentinel-5P TROPOMI, via NASA GES DISC.

Unlike every other layer in this atlas, GES DISC needs a NASA Earthdata
Login bearer token (read from the EARTHDATA_TOKEN environment variable /
GitHub Actions secret - never hardcoded). Without a token this layer is
skipped, not fatal to the build.

TROPOMI is a joint ESA (Sentinel-5P)/NASA product: each granule is one
~5-minute segment of the satellite's orbit, given as a swath (scanline x
ground_pixel), not a lat/lon grid - the granule's own latitude/longitude
arrays are used to place each pixel, then pixels are averaged into this
atlas's own grid. Coverage is whatever the two most recent daytime
overpasses happened to see, so gaps are normal, not a bug.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import tempfile

import numpy as np
import requests

from .geo import CACHE, FRAME, N_LAT, S_LAT, W_LON, E_LON
from .satimg import _png

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION = "C3412185684-GES_DISC"  # S5P_L2__NO2____HiR_NRT v2
BBOX = f"{W_LON},{S_LAT},{E_LON},{N_LAT}"
MAX_GRANULES = 2  # consecutive ~5-min segments of the latest overpass
QA_MIN = 50  # qa_value is 0-100 (scaled from 0-1); NASA recommends >= 50

# mol/m^2 -> molecules/cm^2, the conventional air-quality unit
MOLEC_PER_CM2 = 6.02214e19 / 1e4

# NO2 tropospheric column colour ramp (molecules/cm^2), background -> heavy pollution
STOPS = [
    (2e15, (255, 255, 178, 0)),
    (5e15, (254, 217, 118, 140)),
    (1e16, (254, 178, 76, 180)),
    (2.5e16, (240, 59, 32, 210)),
    (6e16, (149, 0, 44, 235)),
]


def _find_granules():
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(hours=30)
    r = requests.get(CMR_GRANULES, params=dict(
        collection_concept_id=COLLECTION, bounding_box=BBOX,
        temporal=f"{start.isoformat()},{end.isoformat()}",
        page_size=MAX_GRANULES, sort_key="-start_date"), timeout=30)
    r.raise_for_status()
    out = []
    for e in r.json().get("feed", {}).get("entry", []):
        url = next((l["href"] for l in e["links"] if l["href"].endswith(".nc")
                    and "data.gesdisc" in l["href"]), None)
        if url:
            out.append((e["id"], url, e["time_start"]))
    return out


def _download(granule_id, url, token):
    path = os.path.join(CACHE, "s5p", granule_id + ".nc")
    if os.path.exists(path) and dt.datetime.now().timestamp() - os.path.getmtime(path) < 3 * 3600:
        with open(path, "rb") as f:
            return f.read()
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return r.content


def _read_granule(body):
    import h5py
    fd, path = tempfile.mkstemp(suffix=".nc")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(body)
        with h5py.File(path, "r") as f:
            p = f["PRODUCT"]
            lat = p["latitude"][0]
            lon = p["longitude"][0]
            no2 = p["nitrogendioxide_tropospheric_column"][0]
            fill = p["nitrogendioxide_tropospheric_column"].attrs["_FillValue"][0]
            qa = p["qa_value"][0].astype(np.float32)
    finally:
        os.remove(path)
    good = (qa >= QA_MIN) & (no2 < fill * 0.5) & \
        (lat >= S_LAT) & (lat <= N_LAT) & (lon >= W_LON) & (lon <= E_LON)
    return lat[good], lon[good], no2[good] * MOLEC_PER_CM2


def _colorize(grid):
    out = np.zeros(grid.shape + (4,), np.uint8)
    valid = grid > 0
    for i in range(len(STOPS) - 1):
        v0, c0 = STOPS[i]
        v1, c1 = STOPS[i + 1]
        m = valid & (grid >= v0) & (grid < v1)
        t = ((grid[m] - v0) / (v1 - v0))[:, None]
        out[m] = (np.array(c0) * (1 - t) + np.array(c1) * t).astype(np.uint8)
    out[valid & (grid >= STOPS[-1][0])] = STOPS[-1][1]
    return out


def latest():
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        return {"src": None, "pixels": 0, "t": None, "note": "no EARTHDATA_TOKEN set"}
    granules = _find_granules()
    if not granules:
        return {"src": None, "pixels": 0, "t": None}
    lats, lons, vals = [], [], []
    latest_t = None
    for gid, url, t_start in granules:
        body = _download(gid, url, token)
        la, lo, v = _read_granule(body)
        if la.size:
            lats.append(la)
            lons.append(lo)
            vals.append(v)
        if latest_t is None or t_start > latest_t:
            latest_t = t_start
    if not lats:
        return {"src": None, "pixels": 0, "t": latest_t}
    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    val = np.concatenate(vals)

    gx, gy = FRAME.to_grid(lon, lat)
    ix, iy = np.round(gx).astype(int), np.round(gy).astype(int)
    inb = (ix >= 0) & (ix < FRAME.gw) & (iy >= 0) & (iy < FRAME.gh)
    ix, iy, val = ix[inb], iy[inb], val[inb]
    idx = iy * FRAME.gw + ix
    n = FRAME.gw * FRAME.gh
    sums = np.bincount(idx, weights=val, minlength=n)
    counts = np.bincount(idx, minlength=n)
    grid = np.zeros(n, np.float32)
    hit = counts > 0
    grid[hit] = sums[hit] / counts[hit]
    grid = grid.reshape(FRAME.gh, FRAME.gw)

    rgba = _colorize(grid)
    n_px = int(hit.sum())
    t = None
    if latest_t:
        t = int(dt.datetime.fromisoformat(latest_t.replace("Z", "+00:00")).timestamp())
    return {"src": _png(rgba) if n_px else None, "pixels": n_px, "t": t}
