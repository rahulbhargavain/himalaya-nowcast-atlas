"""India's own short-term severe-weather nowcast, straight from IMD's
GeoServer WFS (reactjs.imd.gov.in) - free, no login, no key. District
polygons and station points, refreshed roughly hourly by IMD's forecasters,
each carrying a 4-level colour code (green/yellow/orange/red, IMD's own
national scale) and, whenever a warning is active, IMD's own plain-text
message for the current ~3-hour window - shown verbatim rather than
guessed from the feed's cat1..cat19 flags, whose exact meanings aren't
documented anywhere public.

This is India's official warning product, distinct from every other
hazard layer here (USGS earthquakes, MODIS flood extent, Sentinel-5P air
quality) - none of which are IMD products, and none of which carry an
official "watch/alert/warning" call.

District boundaries as served are survey-grade detailed (tens of
thousands of points for a single district) - far more than a colour tint
needs - so they're simplified with Ramer-Douglas-Peucker before being
turned into SVG paths.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .geo import E_LON, FRAME, N_LAT, S_LAT, W_LON, get_json

WFS = "https://reactjs.imd.gov.in/geoserver/imd/wfs"
PAD = 0.5
SIMPLIFY_DEG = 0.01  # ~1 km at this latitude; plenty for a colour tint
CACHE_S = 1800  # nowcasts update roughly hourly


def _rdp(points, eps):
    """Ramer-Douglas-Peucker polyline simplification (pure numpy - avoids
    adding shapely as a dependency for this one use)."""
    if len(points) < 3:
        return points
    pts = np.asarray(points, dtype=np.float64)
    start, end = pts[0], pts[-1]
    d = end - start
    norm = np.hypot(*d)
    if norm == 0:
        dist = np.hypot(*(pts - start).T)
    else:
        dist = np.abs(d[0] * (start[1] - pts[:, 1]) - (start[0] - pts[:, 0]) * d[1]) / norm
    idx = int(np.argmax(dist))
    if dist[idx] > eps:
        left = _rdp(pts[:idx + 1].tolist(), eps)
        right = _rdp(pts[idx:].tolist(), eps)
        return left[:-1] + right
    return [pts[0].tolist(), pts[-1].tolist()]


def _path(ring):
    simple = _rdp(ring, SIMPLIFY_DEG)
    if len(simple) < 3:
        return ""
    xs, ys = FRAME.to_grid(np.array([c[0] for c in simple]), np.array([c[1] for c in simple]))
    pts = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)]
    dedup = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if p != a]
    return "M" + "L".join(dedup) + "Z" if len(dedup) > 2 else ""


def _epoch(s):
    if not s:
        return None
    try:
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _wfs(typename, extra=None):
    j = get_json(WFS, params=dict(
        service="WFS", version="1.1.0", request="GetFeature", typename=typename,
        srsname="EPSG:4326", outputFormat="application/json", **(extra or {})),
        timeout=90, cache_key=f"imd_nowcast/{typename.split(':')[1]}.json", max_age=CACHE_S)
    return (j or {}).get("features", [])


def _common(p):
    return {
        "color": p.get("Color") or 1,
        "message": (p.get("message") or "").strip(),
        "toi": p.get("toi") or "", "vupto": p.get("vupto") or "",
        "mc": (p.get("MC_RMC") or "").replace("mc_", "").replace("_", " ").title(),
        "t": _epoch(p.get("update_time")),
    }


def districts():
    feats = _wfs("imd:NowcastWarningDistrict", dict(
        bbox=f"{S_LAT - PAD},{W_LON - PAD},{N_LAT + PAD},{E_LON + PAD},urn:ogc:def:crs:EPSG::4326"))
    out = []
    for f in feats:
        geom = f.get("geometry")
        if not geom:
            continue
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        d = " ".join(p for p in (_path(poly[0]) for poly in polys) if p)
        if not d:
            continue
        p = f["properties"]
        rec = _common(p)
        rec.update({"d": d, "district": (p.get("District") or "").title(),
                    "state": (p.get("State") or "").title()})
        out.append(rec)
    return out


def stations():
    feats = _wfs("imd:NowcastWarningStation")
    out = []
    for f in feats:
        geom = f.get("geometry")
        if not geom:
            continue
        lon, lat = geom["coordinates"]
        if not (W_LON - PAD <= lon <= E_LON + PAD and S_LAT - PAD <= lat <= N_LAT + PAD):
            continue
        x, y = FRAME.to_grid(lon, lat)
        if not (0 <= x < FRAME.gw and 0 <= y < FRAME.gh):
            continue
        p = f["properties"]
        rec = _common(p)
        rec.update({"x": float(x), "y": float(y), "name": p.get("Station") or ""})
        out.append(rec)
    return out
