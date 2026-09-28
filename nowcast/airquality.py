"""Air quality and aerosols from Sentinel-5P TROPOMI, via NASA GES DISC:
NO2 (traffic/industry), CO (combustion and biomass-burning smoke), and
UV Aerosol Index (dust/smoke, works through thin cloud unlike the gas
retrievals). All three share the same access pattern, just a different
collection and target variable - see SPECIES below.

Unlike every other layer in this atlas, GES DISC needs a NASA Earthdata
Login bearer token (read from the EARTHDATA_TOKEN environment variable /
GitHub Actions secret - never hardcoded). Without a token these layers
are skipped, not fatal to the build.

TROPOMI is a joint ESA (Sentinel-5P)/NASA product: each granule is one
~5-minute segment of the satellite's orbit, given as a swath (scanline x
ground_pixel), not a lat/lon grid - the granule's own latitude/longitude
arrays are used to place each pixel, then pixels are averaged into this
atlas's own grid. Coverage is whatever the two most recent daytime
overpasses happened to see, so gaps are normal, not a bug.
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile

import numpy as np
import requests

from .geo import CACHE, FRAME, N_LAT, S_LAT, W_LON, E_LON
from .satimg import _png

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
BBOX = f"{W_LON},{S_LAT},{E_LON},{N_LAT}"
MAX_GRANULES = 2  # consecutive ~5-min segments of the latest overpass
QA_MIN = 50  # qa_value is 0-100 (scaled from 0-1); NASA recommends >= 50

MOLEC_PER_CM2 = 6.02214e19  # mol/m^2 -> molecules/cm^2 (matches the granules' own
# multiplication_factor_to_convert_to_molecules_percm2 attribute exactly -
# it already folds in the m^2 -> cm^2 conversion, so no extra /1e4 here)

# Each species: which GES DISC collection/variable to read, how to convert
# units, the colour ramp (value -> RGBA, low to high), and whether values
# below the first stop are real "clean" data (min_value) or just noise to
# drop (no min_value - e.g. NO2/CO clean-air columns aren't 0, but aren't
# interesting to draw either).
SPECIES = {
    "no2": dict(
        collection="C3412185684-GES_DISC", var="nitrogendioxide_tropospheric_column",
        factor=MOLEC_PER_CM2, label="NO2",
        stops=[(2e15, (255, 255, 178, 0)), (5e15, (254, 217, 118, 140)), (1e16, (254, 178, 76, 180)),
               (2.5e16, (240, 59, 32, 210)), (6e16, (149, 0, 44, 235))]),
    "co": dict(
        collection="C3412185659-GES_DISC", var="carbonmonoxide_total_column_corrected",
        factor=MOLEC_PER_CM2, label="CO",
        stops=[(1.8e18, (255, 247, 188, 0)), (2.5e18, (254, 196, 79, 140)), (3.5e18, (217, 95, 14, 190)),
               (5e18, (153, 52, 4, 220)), (8e18, (78, 20, 5, 240))]),
    "uvai": dict(
        collection="C3412185639-GES_DISC", var="aerosol_index_354_388",
        factor=1.0, label="UV Aerosol Index", min_value=-5,
        stops=[(0.7, (224, 196, 240, 0)), (1.5, (194, 130, 224, 150)), (3, (152, 60, 200, 195)),
               (6, (106, 13, 150, 225)), (12, (60, 0, 90, 245))]),
}


def _find_granules(collection):
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(hours=30)
    r = requests.get(CMR_GRANULES, params=dict(
        collection_concept_id=collection, bounding_box=BBOX,
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
    # granule_id encodes the exact orbit segment and timestamp, so the file
    # is immutable once published - no freshness window needed, unlike a
    # "latest" URL. A new overpass gets a new id (and path) automatically.
    path = os.path.join(CACHE, "s5p", granule_id + ".nc")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return r.content


def _read_granule(body, var, factor):
    import h5py
    fd, path = tempfile.mkstemp(suffix=".nc")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(body)
        with h5py.File(path, "r") as f:
            p = f["PRODUCT"]
            lat = p["latitude"][0]
            lon = p["longitude"][0]
            val = p[var][0]
            fill = p[var].attrs["_FillValue"][0]
            qa = p["qa_value"][0].astype(np.float32)
    finally:
        os.remove(path)
    good = (qa >= QA_MIN) & (val < fill * 0.5) & \
        (lat >= S_LAT) & (lat <= N_LAT) & (lon >= W_LON) & (lon <= E_LON)
    return lat[good], lon[good], val[good] * factor


def _colorize(grid, stops):
    out = np.zeros(grid.shape + (4,), np.uint8)
    valid = ~np.isnan(grid)
    for i in range(len(stops) - 1):
        v0, c0 = stops[i]
        v1, c1 = stops[i + 1]
        m = valid & (grid >= v0) & (grid < v1)
        t = ((grid[m] - v0) / (v1 - v0))[:, None]
        out[m] = (np.array(c0) * (1 - t) + np.array(c1) * t).astype(np.uint8)
    out[valid & (grid >= stops[-1][0])] = stops[-1][1]
    return out


def _latest_species(spec):
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        return {"src": None, "pixels": 0, "t": None}
    granules = _find_granules(spec["collection"])
    if not granules:
        return {"src": None, "pixels": 0, "t": None}
    lats, lons, vals = [], [], []
    latest_t = None
    for gid, url, t_start in granules:
        body = _download(gid, url, token)
        la, lo, v = _read_granule(body, spec["var"], spec["factor"])
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
    if "min_value" in spec:
        keep = val >= spec["min_value"]
        lat, lon, val = lat[keep], lon[keep], val[keep]

    gx, gy = FRAME.to_grid(lon, lat)
    ix, iy = np.round(gx).astype(int), np.round(gy).astype(int)
    inb = (ix >= 0) & (ix < FRAME.gw) & (iy >= 0) & (iy < FRAME.gh)
    ix, iy, val = ix[inb], iy[inb], val[inb]
    idx = iy * FRAME.gw + ix
    n = FRAME.gw * FRAME.gh
    sums = np.bincount(idx, weights=val, minlength=n)
    counts = np.bincount(idx, minlength=n)
    grid = np.full(n, np.nan, np.float32)
    hit = counts > 0
    grid[hit] = sums[hit] / counts[hit]
    grid = grid.reshape(FRAME.gh, FRAME.gw)

    rgba = _colorize(grid, spec["stops"])
    n_px = int(hit.sum())
    t = None
    if latest_t:
        t = int(dt.datetime.fromisoformat(latest_t.replace("Z", "+00:00")).timestamp())
    return {"src": _png(rgba) if n_px else None, "pixels": n_px, "t": t}


def latest(species="no2"):
    return _latest_species(SPECIES[species])


def latest_all():
    return {key: _latest_species(spec) for key, spec in SPECIES.items()}
