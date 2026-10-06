"""Recent earthquakes in the Himalayan arc, from the USGS FDSNWS event API
(earthquake.usgs.gov) — free, no key, no auth. India's own National Center
for Seismology (seismo.gov.in) has no public API of this kind; its data is
only browsable through its own portal/app, so USGS's global catalogue is
used instead, restricted to the atlas's bounding box.
"""
from __future__ import annotations

import datetime as dt

from .geo import E_LON, FRAME, N_LAT, S_LAT, W_LON, get_json

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
DAYS = 30
MIN_MAGNITUDE = 3.0


def earthquakes(days=DAYS, min_magnitude=MIN_MAGNITUDE, box=None, frame=FRAME):
    """box = (w, s, e, n) and its frame for a regional page outside the
    atlas's map; default the atlas's own."""
    w, s, e, n = box or (W_LON, S_LAT, E_LON, N_LAT)
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    j = get_json(USGS, params=dict(
        format="geojson", starttime=start.date().isoformat(), endtime=end.date().isoformat(),
        minlatitude=s, maxlatitude=n, minlongitude=w, maxlongitude=e,
        minmagnitude=min_magnitude, orderby="time"), timeout=60)
    out = []
    for f in (j or {}).get("features", []):
        lon, lat, depth = f["geometry"]["coordinates"]
        x, y = frame.to_grid(lon, lat)
        if box is None and not (0 <= x < frame.gw and 0 <= y < frame.gh):
            continue
        a = f["properties"]
        out.append({
            "x": float(x), "y": float(y), "lat": round(lat, 3), "lon": round(lon, 3),
            "mag": a.get("mag"), "depth_km": round(depth, 1) if depth is not None else None,
            "place": a.get("place"), "t": int(a["time"] / 1000) if a.get("time") else None,
            "id": f.get("id"),
        })
    return out
