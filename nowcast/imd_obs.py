"""IMD ground and radar observations, all from IMD's public endpoints (no
login, no key):

  * AWS/ARG: automatic weather stations and rain gauges, from the GeoServer
    WFS layer imd:aws_data_layer. Many stations stop reporting for months,
    so only reports newer than `max_age_h` are kept.
  * SYNOP: the staffed observatories (3-hourly), imd:synop_data_layer.
  * IMD's Doppler weather radars in the Uttarakhand hills (Mukteshwar,
    Lansdowne, Surkanda Devi): each one's latest MAX(Z) product image
    (mausam.imd.gov.in/Radar/caz_<code>.gif), decoded back to dBZ and
    georeferenced so it can sit on the map under the other layers.

The radar images have no georeferencing of their own. Mukteshwar's fit was
made from 16 town markers IMD draws on the image (RMS error about 0.8 km)
and places the radar at 29.459N 79.660E. The fit shows the plan view is a
plain local projection: the radar at the centre of the plan view, 100 km to
its edge, north up. The other radars are placed by that geometry from their
sites and plan-view boxes, checked against their own town markers (see
RADARS). If IMD changes an image's layout the size check fails and only the
original image is shown.
"""
from __future__ import annotations

import base64
import datetime as dt
import email.utils
import io

import cv2
import numpy as np
from PIL import Image

from .geo import _session, get_json

WFS = "https://reactjs.imd.gov.in/geoserver/imd/wfs"
MKS_SITE = (29.459, 79.660)
# lon, lat, 1 -> image x, y (least squares on IMD's own town markers).
MKS_AFFINE = np.array([[2.89784983e+02, 6.95062116e-01],
                       [-1.97551599e-01, -3.34207955e+02],
                       [-2.27770625e+04, 1.02781044e+04]])


def _geometric_affine(site, plan, range_km=100):
    """lon, lat, 1 -> image x, y for a plan view centred on the radar with
    `range_km` from centre to edge (Mukteshwar's fit agrees to 0.4%)."""
    lat0, lon0 = site
    x0, y0, x1, y1 = plan
    k = (x1 - x0) / 2 / range_km  # px per km
    kx = k * 111.32 * np.cos(np.radians(lat0))
    ky = k * 110.95
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.array([[kx, 0.0], [0.0, -ky], [cx - kx * lon0, cy + ky * lat0]])


# code -> name, site (lat, lon), image size, plan view box (x0, y0, x1, y1),
# and the lon/lat -> pixel fit. Sites are IMD's own (its radar page's map).
RADARS = {
    "mks": {"name": "Mukteshwar", "site": MKS_SITE, "size": (1017, 787), "plan": (0, 187, 600, 787),
            "affine": MKS_AFFINE},
    "ldn": {"name": "Lansdowne", "site": (29.85, 78.6767), "size": (1017, 787), "plan": (0, 187, 600, 787)},
    "sur": {"name": "Surkanda Devi", "site": (30.4117, 78.2883), "size": (1272, 1042), "plan": (0, 242, 800, 1042)},
}
for _r in RADARS.values():
    _r.setdefault("affine", _geometric_affine(_r["site"], _r["plan"]))
RANGE_KM = 100
RADAR_URL = "https://mausam.imd.gov.in/Radar/caz_{code}.gif"

# IMD's MAX(Z) legend: lower bound of each bin (dBZ) and its colour.
LEGEND = [(20.0, (0, 0, 127)), (22.7, (0, 0, 160)), (25.3, (0, 0, 199)), (28.0, (0, 52, 255)),
              (30.7, (0, 121, 255)), (33.3, (26, 162, 255)), (36.0, (83, 208, 255)), (38.7, (135, 240, 255)),
              (41.3, (255, 255, 255)), (44.0, (255, 248, 196)), (46.7, (255, 229, 1)), (49.3, (254, 188, 0)),
              (52.0, (250, 111, 4)), (54.7, (255, 63, 0)), (57.3, (199, 1, 0))]


def _wfs_all(typename):
    j = get_json(WFS, params=dict(service="WFS", version="1.1.0", request="GetFeature", typename=typename,
                                  srsname="EPSG:4326", outputFormat="application/json"), timeout=120)
    return (j or {}).get("features", [])


def _num(v):
    if v in (None, "", "NULL", "null"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _in(geom, box):
    if not geom:
        return None
    lon, lat = geom["coordinates"][:2]
    w, s, e, n = box
    return (lon, lat) if w <= lon <= e and s <= lat <= n else None


def aws(box, frame, max_age_h=3):
    now = dt.datetime.now(dt.timezone.utc)
    best = {}
    silent = 0
    for f in _wfs_all("imd:aws_data_layer"):
        ll = _in(f.get("geometry"), box)
        if not ll:
            continue
        p = f["properties"]
        try:
            day = dt.date.fromisoformat((p.get("dat") or "")[:10])
            hh, mm = (p.get("time") or "")[11:16].split(":")
            t = dt.datetime(day.year, day.month, day.day, int(hh), int(mm), tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if now - t > dt.timedelta(hours=max_age_h):
            silent += 1
            continue
        name = (p.get("station") or "").replace("_", " ").title()
        key = round(ll[0], 2), round(ll[1], 2)
        if key in best and best[key]["t"] >= t.timestamp():
            continue
        x, y = frame.to_grid(*ll)
        best[key] = {"name": name, "lon": ll[0], "lat": ll[1], "x": float(x), "y": float(y),
                     "t": int(t.timestamp()), "temp": _num(p.get("temp")), "tmin": _num(p.get("temp_min")),
                     "tmax": _num(p.get("temp_max")), "rh": _num(p.get("rh")), "rain": _num(p.get("rainfall")),
                     "wdir": _num(p.get("winddir")), "wspd": _num(p.get("windspeed")),
                     "kind": "ARG" if all(_num(p.get(k)) is None for k in ("temp", "rh")) else "AWS"}
    return sorted(best.values(), key=lambda s: s["name"]), silent


def synop(box, frame, max_age_h=6):
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for f in _wfs_all("imd:synop_data_layer"):
        ll = _in(f.get("geometry"), box)
        if not ll:
            continue
        p = f["properties"]
        try:
            day = dt.date.fromisoformat((p.get("dat") or "")[:10])
            t = dt.datetime(day.year, day.month, day.day, int(p.get("utc") or 0), tzinfo=dt.timezone.utc)
        except (ValueError, TypeError):
            continue
        if now - t > dt.timedelta(hours=max_age_h):
            continue
        x, y = frame.to_grid(*ll)
        out.append({"name": p.get("station") or str(p.get("station_id")), "id": p.get("station_id"),
                    "lon": ll[0], "lat": ll[1], "x": float(x), "y": float(y), "t": int(t.timestamp()),
                    "temp": _num(p.get("dbtemp")), "dew": _num(p.get("dewtemp")), "rh": _num(p.get("rh")),
                    "mslp": _num(p.get("mslp")), "dp24": _num(p.get("pressure_24hr_change")),
                    "wdir": _num(p.get("winddir")), "wspd": _num(p.get("windsp")), "ww": _num(p.get("weather")),
                    "cloud": _num(p.get("nebulosity")), "vis": _num(p.get("visibility")),
                    "rain3": _num(p.get("3hrlyrain")), "rain24": _num(p.get("24hrlyrain")),
                    "tmax": _num(p.get("max")), "tmin": _num(p.get("min"))})
    return sorted(out, key=lambda s: s["name"])


def _decode(a, plan):
    """IMD MAX(Z) plan view -> dBZ (float32, NaN where no echo).

    IMD also draws range rings, labels and town dots in some of the legend's
    colours (the whites, creams and reds). Two filters drop those:
      1. high-reflectivity pixels (41 dBZ and up) are accepted bin by bin,
         upward: each needs an already-accepted neighbour one or two bins
         lower. Real storm cores grade up through the blues; overlays jump
         straight from background to white or red;
      2. every blob must hold some low-reflectivity blues, which the
         background never contains."""
    x0, y0, x1, y1 = plan
    pv = a[y0:y1, x0:x1].astype(np.int32)
    packed = (pv[..., 0] << 16) | (pv[..., 1] << 8) | pv[..., 2]
    dbz = np.full(packed.shape, np.nan, np.float32)
    bins = np.full(packed.shape, -1, np.int16)
    low = np.zeros(packed.shape, bool)
    for k, (v, (r, g, b)) in enumerate(LEGEND):
        m = packed == ((r << 16) | (g << 8) | b)
        dbz[m] = v
        bins[m] = k
        if v < 41:
            low |= m
    ok = bins.copy()
    ok[bins >= 8] = -1
    k5 = np.ones((5, 5), np.uint8)
    for k in range(8, len(LEGEND)):
        lower = ((ok == k - 1) | (ok == k - 2)).astype(np.uint8)
        grow = (bins == k) & cv2.dilate(lower, k5).astype(bool)
        # Let a core spread within its own bin once it has a foothold.
        for _ in range(40):
            nxt = (bins == k) & cv2.dilate(grow.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if (nxt == grow).all():
                break
            grow = nxt
        ok[grow] = k
    # Snow on IMD's terrain backdrop is the same white and cream as 41-44 dBZ,
    # and can touch real echo. A real core is mostly ringed by echo; a snow
    # patch at the edge of a rain band is mostly ringed by terrain.
    cand = (ok >= 8).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    for i in range(1, n):
        blob = lab == i
        ring = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~blob
        if ring.any() and (bins[ring] >= 0).mean() < 0.6:
            ok[blob] = -1
    dbz[(bins >= 8) & (ok < 8)] = np.nan
    echo = np.isfinite(dbz)
    n, lab = cv2.connectedComponents(cv2.dilate(echo.astype(np.uint8), np.ones((3, 3), np.uint8)), connectivity=8)
    lows = np.bincount(lab[low], minlength=n)
    keep = lows >= 4
    keep[0] = False
    dbz[~keep[lab]] = np.nan
    yy, xx = np.mgrid[0:pv.shape[0], 0:pv.shape[1]]
    r = (x1 - x0) / 2
    dbz[(xx - r) ** 2 + (yy - r) ** 2 > (r - 2) ** 2] = np.nan
    return dbz


def radar(code, frame):
    """Latest MAX(Z) from IMD radar `code` (see RADARS): original image (data
    URI), the image time (HTTP Last-Modified), and the decoded dBZ resampled
    onto `frame`'s grid."""
    R = RADARS[code]
    r = _session.get(RADAR_URL.format(code=code), timeout=60)
    r.raise_for_status()
    t = None
    if r.headers.get("Last-Modified"):
        t = int(email.utils.parsedate_to_datetime(r.headers["Last-Modified"]).timestamp())
    im = Image.open(io.BytesIO(r.content))
    cx, cy = frame.to_grid(R["site"][1], R["site"][0])
    out = {"code": code, "name": R["name"], "t": t, "site": R["site"], "range_km": RANGE_KM,
           "image": "data:image/gif;base64," + base64.b64encode(r.content).decode(), "dbz": None,
           "x": float(cx), "y": float(cy), "r_px": RANGE_KM / frame.km_per_grid_px}
    if im.size != R["size"]:
        out["note"] = f"image size {im.size} differs from the layout this decoder was fitted to"
        return out
    plan = R["plan"]
    pv = _decode(np.asarray(im.convert("RGB")), plan)
    lon, lat = frame.grid_lonlat()
    A = R["affine"]
    px = lon * A[0, 0] + lat * A[1, 0] + A[2, 0] - plan[0]
    py = lon * A[0, 1] + lat * A[1, 1] + A[2, 1] - plan[1]
    grid = cv2.remap(np.nan_to_num(pv, nan=-1).astype(np.float32), px.astype(np.float32), py.astype(np.float32),
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    out["dbz"] = np.where(grid > 0, grid, 0).astype(np.float32)
    out["echo_px"] = int((out["dbz"] > 0).sum())
    return out
