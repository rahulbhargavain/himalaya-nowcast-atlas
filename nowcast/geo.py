"""Map frame, projection and HTTP helpers shared by every layer.

Everything is drawn on one Web Mercator pixel grid so layers line up
without any client-side reprojection:

  * FULL  = zoom 7 tile pixels (~1.06 km/px at 30N) -- hillshade only
  * GRID  = FULL / 2 (~2.1 km/px) -- DEM, land cover, radar, snow, overlays

The browser draws GRID-sized canvases stretched over the FULL-size
hillshade, so a GRID pixel (x, y) sits at FULL pixel (2x, 2y).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import io
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image

W_LON, E_LON, S_LAT, N_LAT = 72.0, 97.0, 26.0, 37.0
ZOOM = 7
TILE = 256
R_EARTH = 6378137.0

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

_session = requests.Session()
_session.headers["User-Agent"] = "himalaya-nowcast/1.0 (personal weather visualisation)"


def world_px(lon, lat, z=ZOOM):
    """Lon/lat (degrees, scalars or arrays) -> global Web Mercator pixel coords at zoom z."""
    n = TILE * 2 ** z
    lat = np.clip(lat, -85.0, 85.0)
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    r = np.radians(lat)
    y = (1.0 - np.log(np.tan(r) + 1.0 / np.cos(r)) / math.pi) / 2.0 * n
    return x, y


def world_lonlat(x, y, z=ZOOM):
    n = TILE * 2 ** z
    lon = np.asarray(x) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * np.asarray(y) / n))))
    return lon, lat


@dataclass(frozen=True)
class Frame:
    x0: int  # FULL-res global pixel origin
    y0: int
    w: int   # FULL-res size
    h: int
    z: int = ZOOM  # FULL-res zoom level; GRID is one level coarser

    @property
    def gw(self):
        return self.w // 2

    @property
    def gh(self):
        return self.h // 2

    @property
    def km_per_grid_px(self):
        """Approximate GRID pixel size at the frame's middle latitude."""
        lat = world_lonlat(self.x0, self.y0 + self.h / 2, self.z)[1]
        return 2 * 156.54303392 * math.cos(math.radians(float(lat))) / 2 ** self.z

    def to_grid(self, lon, lat):
        """Lon/lat -> GRID pixel coords (float)."""
        x, y = world_px(lon, lat, self.z)
        return (np.asarray(x) - self.x0) / 2.0, (np.asarray(y) - self.y0) / 2.0

    def grid_lonlat(self):
        """Lon/lat of every GRID pixel centre, shape (gh, gw)."""
        xs = self.x0 + (np.arange(self.gw) + 0.5) * 2.0
        ys = self.y0 + (np.arange(self.gh) + 0.5) * 2.0
        lon, _ = world_lonlat(xs, np.full_like(xs, ys[0]), self.z)
        _, lat = world_lonlat(np.full_like(ys, xs[0]), ys, self.z)
        return np.meshgrid(lon, lat)

    def merc_bbox(self):
        """EPSG:3857 bbox (minx, miny, maxx, maxy) in metres."""
        n = TILE * 2 ** self.z
        half = math.pi * R_EARTH
        def mx(px):
            return px / n * 2 * half - half
        def my(py):
            return half - py / n * 2 * half
        return mx(self.x0), my(self.y0 + self.h), mx(self.x0 + self.w), my(self.y0)

    def tag(self):
        """Cache-key suffix: empty for the main frame, so its cache is unchanged."""
        return "" if self == FRAME else f"_{self.z}_{self.x0}_{self.y0}_{self.w}x{self.h}"


def make_frame(w_lon=W_LON, e_lon=E_LON, s_lat=S_LAT, n_lat=N_LAT, z=ZOOM) -> Frame:
    x0, y0 = world_px(w_lon, n_lat, z)
    x1, y1 = world_px(e_lon, s_lat, z)
    x0, y0 = int(math.floor(x0)) // 2 * 2, int(math.floor(y0)) // 2 * 2
    w = (int(math.ceil(x1)) - x0) // 4 * 4
    h = (int(math.ceil(y1)) - y0) // 4 * 4
    return Frame(x0, y0, w, h, z)


FRAME = make_frame()


def get(url, params=None, timeout=60, tries=6, cache_key=None, max_age=None, stale_ok=False, verify=True):
    """GET with retries. With cache_key, the body is cached on disk
    (forever, or for max_age seconds). With stale_ok, an expired copy is
    returned if the fetch fails - for slow-refresh pages, where last month's
    data beats no page; never for live layers."""
    path = None
    if cache_key:
        path = os.path.join(CACHE, cache_key)
        if os.path.exists(path) and (max_age is None or time.time() - os.path.getmtime(path) < max_age):
            with open(path, "rb") as f:
                return f.read()
    last = None
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=timeout, verify=verify)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                # Rate-limited or a transient upstream outage (Open-Meteo has
                # returned occasional 503s): worth waiting out, not bailing.
                last = f"{r.status_code} {r.reason}"
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(r.content)
            return r.content
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    if stale_ok and path and os.path.exists(path):
        print(f"  GET failed, using the cached copy from {time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(path)))}: "
              f"{url} ({last})", flush=True)
        with open(path, "rb") as f:
            return f.read()
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")


def get_json(url, params=None, tries=3, **kw):
    """Like get(), but also retries if the upstream returns a
    truncated/malformed body (seen occasionally from Open-Meteo)."""
    import json
    last = None
    for attempt in range(tries):
        body = get(url, params=params, **kw)
        if body is None:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Malformed JSON after {tries} tries: {url} ({last})")


def fetch_many(fn, items, workers=8):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def stitch_tiles(url_fmt, z, frame_px_box, cache_prefix=None, workers=8, mode="RGBA"):
    """Fetch every z-level tile covering the global pixel box (x0, y0, x1, y1)
    at zoom z, stitch, and crop to the box. Missing tiles stay transparent."""
    x0, y0, x1, y1 = frame_px_box
    tx0, ty0 = int(x0 // TILE), int(y0 // TILE)
    tx1, ty1 = int((x1 - 1) // TILE), int((y1 - 1) // TILE)
    tiles = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

    def one(t):
        tx, ty = t
        key = None
        if cache_prefix:
            key = f"{cache_prefix}/{z}_{tx}_{ty}.png"
        body = get(url_fmt.format(z=z, x=tx, y=ty), cache_key=key)
        if not body:
            return t, None
        return t, Image.open(io.BytesIO(body)).convert(mode)

    canvas = Image.new(mode, ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    for (tx, ty), im in fetch_many(one, tiles, workers):
        if im is not None:
            canvas.paste(im, ((tx - tx0) * TILE, (ty - ty0) * TILE))
    ox, oy = int(round(x0 - tx0 * TILE)), int(round(y0 - ty0 * TILE))
    return canvas.crop((ox, oy, ox + int(x1 - x0), oy + int(y1 - y0)))


def sample_latlon_grid(values, lats, lons, frame=FRAME):
    """Bilinearly sample a regular lat/lon grid (values[i_lat, j_lon],
    lats ascending) at every GRID pixel. Returns (gh, gw) float32."""
    import cv2
    lon, lat = frame.grid_lonlat()
    fi = (lat - lats[0]) / (lats[1] - lats[0])
    fj = (lon - lons[0]) / (lons[1] - lons[0])
    return cv2.remap(values.astype(np.float32), fj.astype(np.float32), fi.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]
