"""India's own short-term severe-weather nowcast and 5-day district
warnings, straight from IMD's GeoServer WFS (reactjs.imd.gov.in) - free,
no login, no key.

Nowcast: district polygons and station points, refreshed roughly hourly by
IMD's forecasters, each with a colour (1 green ... 4 red) and the active
hazard categories (cat1..cat19). The category meanings and colour scale are
IMD's own, from the field reference at api.imd.gov.in/public/api_reference.html.
When a forecaster adds free text it is shown verbatim as well.

District warnings: IMD's 5-day district-wise warnings (warning codes per
day). Their colour field runs the other way (1 red ... 4 green), so both are
normalised here to one severity scale: 1 none, 2 yellow, 3 orange, 4 red.

District boundaries as served are survey-grade detailed (tens of thousands
of points for a single district) - far more than a colour tint needs - so
they're simplified with Ramer-Douglas-Peucker before being turned into SVG
paths.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .geo import E_LON, FRAME, N_LAT, S_LAT, W_LON, get_json

WFS = "https://reactjs.imd.gov.in/geoserver/imd/wfs"
PAD = 0.5
SIMPLIFY_DEG = 0.01  # ~1 km at this latitude; plenty for a colour tint
CACHE_S = 1800  # nowcasts update roughly hourly
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# Nowcast categories: field -> (value IMD stores when active, meaning).
NOWCAST_CATS = {
    "cat2": "Light rain (<5 mm/h)", "cat3": "Light snow (<5 cm/h)",
    "cat4": "Light thunderstorm (gusts <40 km/h)", "cat5": "Slight dust storm",
    "cat6": "Lightning, low probability (<30%)", "cat7": "Moderate rain (5-15 mm/h)",
    "cat8": "Moderate snow (5-15 cm/h)", "cat9": "Moderate thunderstorm (gusts 41-61 km/h)",
    "cat10": "Moderate dust storm", "cat11": "Lightning, moderate probability (30-60%)",
    "cat12": "Heavy rain (>15 mm/h)", "cat13": "Heavy snow (>15 cm/h)",
    "cat14": "Severe thunderstorm (gusts 62-87 km/h)", "cat15": "Very severe thunderstorm (gusts >87 km/h)",
    "cat17": "Thunderstorm with hail", "cat18": "Severe dust storm",
    "cat19": "Lightning, high probability (>60%)",
}

# 5-day district warning codes.
WARNING_CODES = {
    2: "Heavy rain", 3: "Heavy snow", 4: "Thunderstorm & lightning, squall", 5: "Hailstorm",
    6: "Dust storm", 7: "Dust-raising winds", 8: "Strong surface winds", 9: "Heat wave",
    10: "Hot day", 11: "Warm night", 12: "Cold wave", 13: "Cold day", 14: "Ground frost",
    15: "Fog", 16: "Very heavy rain", 17: "Extremely heavy rain",
}


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


def _path(ring, frame, eps):
    simple = _rdp(ring, eps)
    if len(simple) < 3:
        return ""
    xs, ys = frame.to_grid(np.array([c[0] for c in simple]), np.array([c[1] for c in simple]))
    pts = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)]
    dedup = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if p != a]
    return "M" + "L".join(dedup) + "Z" if len(dedup) > 2 else ""


def _epoch(s):
    """IMD's update stamps end in "Z" but are IST: a nowcast updated at
    "19:12Z" shows up in a build made at 16:12 UTC."""
    if not s:
        return None
    try:
        return int(dt.datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=IST).timestamp())
    except ValueError:
        return None


def _wfs(typename, extra=None, key=None):
    j = get_json(WFS, params=dict(
        service="WFS", version="1.1.0", request="GetFeature", typename=typename,
        srsname="EPSG:4326", outputFormat="application/json", **(extra or {})),
        timeout=90, cache_key=f"imd_nowcast/{key or typename.split(':')[1]}.json", max_age=CACHE_S)
    return (j or {}).get("features", [])


def _hazards(p):
    out = [name for key, name in NOWCAST_CATS.items() if p.get(key) not in (None, 0, "", "0")]
    extra = (p.get("cat16") or "").strip()
    if extra:
        out.append(extra)
    return out


def _common(p):
    return {
        "color": p.get("Color") or 1,
        "hazards": _hazards(p),
        "message": (p.get("message") or "").strip(),
        "toi": p.get("toi") or "", "vupto": p.get("vupto") or "",
        "mc": (p.get("MC_RMC") or "").replace("mc_", "").replace("_", " ").title(),
        "t": _epoch(p.get("update_time") or p.get("updated_at")),
    }


def _ring_box(polys):
    xs = [c[0] for poly in polys for c in poly[0]]
    ys = [c[1] for poly in polys for c in poly[0]]
    return min(xs), min(ys), max(xs), max(ys)


def _overlaps(b, box):
    w, s, e, n = box
    return not (b[2] < w or b[0] > e or b[3] < s or b[1] > n)


def districts(frame=FRAME, box=None, eps=SIMPLIFY_DEG):
    """box = (w, s, e, n) to keep only districts touching it (defaults to the
    main map). Always fetched for the main map's box, so other maps inside it
    reuse the same cached response."""
    if box and not (box[0] >= W_LON - PAD and box[1] >= S_LAT - PAD and box[2] <= E_LON + PAD
                    and box[3] <= N_LAT + PAD):
        w, s, e, n = box
        feats = _wfs("imd:NowcastWarningDistrict", dict(
            bbox=f"{s - PAD},{w - PAD},{n + PAD},{e + PAD},urn:ogc:def:crs:EPSG::4326"),
            key=f"NowcastWarningDistrict_{w}_{s}")
    else:
        feats = _wfs("imd:NowcastWarningDistrict", dict(
            bbox=f"{S_LAT - PAD},{W_LON - PAD},{N_LAT + PAD},{E_LON + PAD},urn:ogc:def:crs:EPSG::4326"))
    out = []
    for f in feats:
        geom = f.get("geometry")
        if not geom:
            continue
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        if box and not _overlaps(_ring_box(polys), box):
            continue
        d = " ".join(p for p in (_path(poly[0], frame, eps) for poly in polys) if p)
        if not d:
            continue
        p = f["properties"]
        rec = _common(p)
        rec.update({"d": d, "district": (p.get("District") or "").title(),
                    "state": (p.get("State") or "").title()})
        out.append(rec)
    return out


def stations(frame=FRAME, box=None):
    feats = _wfs("imd:NowcastWarningStation")
    w, s, e, n = box or (W_LON - PAD, S_LAT - PAD, E_LON + PAD, N_LAT + PAD)
    out = []
    for f in feats:
        geom = f.get("geometry")
        if not geom:
            continue
        lon, lat = geom["coordinates"]
        if not (w <= lon <= e and s <= lat <= n):
            continue
        x, y = frame.to_grid(lon, lat)
        if not (0 <= x < frame.gw and 0 <= y < frame.gh):
            continue
        p = f["properties"]
        rec = _common(p)
        rec.update({"x": float(x), "y": float(y), "name": p.get("Station") or "", "lat": lat, "lon": lon})
        out.append(rec)
    return out


def district_warnings(names):
    """IMD's 5-day warnings for the named districts (upper-case, as IMD spells
    them). Severity is normalised to 1 none, 2 yellow, 3 orange, 4 red."""
    fields = ["District", "Date", "UTC", "updated_at"] + [f"Day_{k}" for k in range(1, 6)] + \
        [f"Day{k}_Color" for k in range(1, 6)] + [f"Day{k}_text" for k in range(1, 6)]
    feats = _wfs("imd:district_warnings_india", dict(propertyName=",".join(fields)))
    want = {n.upper() for n in names}
    out = {}
    for f in feats:
        p = f["properties"]
        name = (p.get("District") or "").upper()
        if name not in want:
            continue
        try:
            issued = dt.date.fromisoformat(p["Date"][:10])
        except (KeyError, ValueError):
            continue
        days = []
        for k in range(1, 6):
            codes = [int(c) for c in str(p.get(f"Day_{k}") or "").split(",") if c.strip().isdigit()]
            col = p.get(f"Day{k}_Color")
            sev = 5 - int(col) if col in (1, 2, 3, 4) else 1
            days.append({"date": (issued + dt.timedelta(days=k - 1)).isoformat(), "severity": sev,
                         "hazards": [WARNING_CODES[c] for c in codes if c in WARNING_CODES],
                         "text": (p.get(f"Day{k}_text") or "").strip()})
        prev = out.get(name)
        if prev is None or (p.get("updated_at") or "") > prev["_upd"]:
            out[name] = {"district": name.title(), "issued": issued.isoformat(), "utc": p.get("UTC"),
                         "t": _epoch(p.get("updated_at")), "days": days, "_upd": p.get("updated_at") or ""}
    for v in out.values():
        v.pop("_upd")
    return [out[n] for n in sorted(out)]
