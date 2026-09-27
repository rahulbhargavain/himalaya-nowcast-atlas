"""Satellite-observed rainfall: NASA/JAXA GPM IMERG near-real-time, via
NASA GIBS's public WMS (the same free, no-login endpoint the VIIRS fire
layer already uses). This is real multi-satellite-observed precipitation,
distinct from the model's forecast rain and the ground radar mosaic.

Full-resolution/archival IMERG (via GES DISC) needs a free NASA Earthdata
Login; JAXA's own GSMaP near-real-time feed needs a separate registration
too. GIBS's preview tiles need neither, at the cost of coarser (~10 km,
JPEG/PNG-quantised) resolution - fine for a glanceable overlay, not for
quantitative work.
"""
from __future__ import annotations

import base64
import datetime as dt
import io

import numpy as np
from PIL import Image

from .geo import FRAME, get
from .land import GIBS_WMS

LAYER = "IMERG_Precipitation_Rate_30min"


def _png(arr):
    im = Image.fromarray(arr, "RGBA")
    if (arr[..., 3] > 0).mean() > 0.02:
        im = im.quantize(colors=96, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def latest(max_back_hours=6):
    """The most recent available IMERG frame within max_back_hours. IMERG
    near-real-time typically lags several hours, so this steps back in
    30-minute slots until it finds one with data."""
    minx, miny, maxx, maxy = FRAME.merc_bbox()
    now = dt.datetime.now(dt.timezone.utc)
    slot = now.replace(minute=30 if now.minute >= 30 else 0, second=0, microsecond=0)
    for back in range(0, max_back_hours * 2 + 1):
        t = slot - dt.timedelta(minutes=30 * back)
        body = get(GIBS_WMS, params=dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0",
                                         LAYERS=LAYER, STYLES="", CRS="EPSG:3857",
                                         BBOX=f"{minx},{miny},{maxx},{maxy}", WIDTH=FRAME.gw, HEIGHT=FRAME.gh,
                                         FORMAT="image/png", TRANSPARENT="TRUE",
                                         TIME=t.strftime("%Y-%m-%dT%H:%M:00Z")), timeout=60)
        if not body:
            continue
        a = np.asarray(Image.open(io.BytesIO(body)).convert("RGBA"))
        if (a[..., 3] > 0).any():
            return {"src": _png(a), "t": int(t.replace(tzinfo=dt.timezone.utc).timestamp())}
    return {"src": None, "t": None}
