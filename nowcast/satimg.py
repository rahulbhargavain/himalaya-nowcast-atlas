"""Two more NASA/JAXA satellite views, both via NASA GIBS's free, no-login
WMS (the same endpoint the fire and IMERG-rain layers already use):

  * True colour: an actual daily satellite photo (MODIS Terra Corrected
    Reflectance), for real cloud cover / snow extent / haze context that
    neither the radar mosaic nor IMERG rain rate can show.
  * Flood extent: MODIS's Near-Real-Time Global Flood Mapping product,
    observed surface water/flood, distinct from the GloFAS *modelled*
    river discharge already shown for named gauges. Most of the frame is
    "Insufficient Data" grey - that class is dropped so only the
    meaningful classes (surface water, recurring flood, flood) show.

Both layers default to the latest available date automatically when no
TIME parameter is given, so no fallback stepping is needed (unlike IMERG,
whose near-real-time granules lag the WMS's advertised default).
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .geo import FRAME, get
from .land import GIBS_WMS

TRUE_COLOR_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
FLOOD_LAYER = "MODIS_Combined_Flood_1-Day"

# Water Product legend: cyan=Surface Water, yellow=Recurring Flood,
# red=Flood, grey=Insufficient Data (dropped).
FLOOD_CLASSES = [
    ((50, 210, 245), (70, 170, 220, 200)),   # surface water -> softer blue
    ((255, 255, 0), (235, 180, 20, 220)),    # recurring flood -> amber
    ((250, 30, 36), (214, 40, 40, 235)),     # flood -> red
]


def _png(arr):
    im = Image.fromarray(arr, "RGBA")
    im = im.quantize(colors=32, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def true_color():
    minx, miny, maxx, maxy = FRAME.merc_bbox()
    body = get(GIBS_WMS, params=dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0",
                                     LAYERS=TRUE_COLOR_LAYER, STYLES="", CRS="EPSG:3857",
                                     BBOX=f"{minx},{miny},{maxx},{maxy}", WIDTH=FRAME.gw, HEIGHT=FRAME.gh,
                                     FORMAT="image/jpeg", TRANSPARENT="FALSE"), timeout=90)
    if not body:
        return {"src": None}
    return {"src": "data:image/jpeg;base64," + base64.b64encode(body).decode()}


def flood_extent():
    minx, miny, maxx, maxy = FRAME.merc_bbox()
    body = get(GIBS_WMS, params=dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0",
                                     LAYERS=FLOOD_LAYER, STYLES="", CRS="EPSG:3857",
                                     BBOX=f"{minx},{miny},{maxx},{maxy}", WIDTH=FRAME.gw, HEIGHT=FRAME.gh,
                                     FORMAT="image/png", TRANSPARENT="TRUE"), timeout=90)
    if not body:
        return {"src": None, "pixels": 0}
    a = np.asarray(Image.open(io.BytesIO(body)).convert("RGBA")).copy()
    out = np.zeros_like(a)
    hit = np.zeros(a.shape[:2], bool)
    for (r, g, b), rgba in FLOOD_CLASSES:
        m = (a[..., 0] == r) & (a[..., 1] == g) & (a[..., 2] == b)
        out[m] = rgba
        hit |= m
    n = int(hit.sum())
    return {"src": _png(out) if n else None, "pixels": n}
