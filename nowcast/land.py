"""Static and slow-changing surface layers: terrain, land cover, tree/shrub
line, observed snow (recent days and monthly), and the seasonal snow line
per sector derived from it.

Sources (all free, no key):
  * Terrain: AWS Terrain Tiles (terrarium encoding, SRTM/GMTED blend).
  * Land cover: MODIS MCD12Q1 IGBP classes via NASA GIBS WMS (500 m).
  * Snow: MODIS Terra NDSI snow cover (daily) and monthly average snow
    cover percent via NASA GIBS WMS.
"""
from __future__ import annotations

import datetime as dt
import io
import re

import cv2
import numpy as np
from PIL import Image

from .geo import CACHE, FRAME, get, stitch_tiles

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"

# IGBP class ids (MCD12Q1 LC_Type1) -> short names used by the page.
IGBP = {
    1: "Evergreen needleleaf forest", 2: "Evergreen broadleaf forest", 3: "Deciduous needleleaf forest",
    4: "Deciduous broadleaf forest", 5: "Mixed forest", 6: "Closed shrubland", 7: "Open shrubland",
    8: "Woody savanna", 9: "Savanna", 10: "Grassland", 11: "Wetland", 12: "Cropland", 13: "Urban",
    14: "Cropland/natural mosaic", 15: "Permanent snow and ice", 16: "Barren", 17: "Water",
}
FOREST = {1, 2, 3, 4, 5, 8}      # 8 (woody savanna) is open forest along the Himalayan treeline
SHRUB = {6, 7}
OPEN_HIGH = {9, 10, 16}          # alpine meadow/steppe, bare rock and scree

# Sectors for the per-sector tree line / snow line statistics.
# Sectors for the per-sector tree line / snow line statistics. Each keeps
# to a latitude band on the Himalayan arc itself, so the dry Tibetan
# plateau behind the crest doesn't dilute the numbers.
SECTORS = [
    ("Karakoram & Kashmir", 72.0, 77.0, 32.5, 36.5),
    ("Himachal & Ladakh", 77.0, 79.0, 30.8, 33.5),
    ("Uttarakhand & far-west Nepal", 79.0, 82.0, 28.8, 31.2),
    ("Central & east Nepal", 82.0, 88.0, 27.3, 28.9),
    ("Sikkim & Bhutan", 88.0, 92.0, 26.8, 28.3),
    ("Arunachal & SE Tibet", 92.0, 97.0, 27.0, 29.8),
]


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------

def load_dem():
    """Elevation (m) at FULL resolution, float32 (h, w)."""
    im = stitch_tiles(
        "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        7, (FRAME.x0, FRAME.y0, FRAME.x0 + FRAME.w, FRAME.y0 + FRAME.h),
        cache_prefix="terrarium", mode="RGB",
    )
    a = np.asarray(im).astype(np.float32)
    dem = a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0
    return np.clip(dem, -100, 9000)


def _pixel_size_m(lat_rows, z=7):
    return 156543.03392 * np.cos(np.radians(lat_rows)) / 2 ** z


def slope_deg(dem, lat_rows, scale=1):
    """Slope in degrees. lat_rows is the latitude of each row; scale is the
    pixel-size multiple relative to FULL (2 for GRID)."""
    px = _pixel_size_m(lat_rows)[:, None] * scale
    gy, gx = np.gradient(dem)
    return np.degrees(np.arctan(np.hypot(gx / px, gy / px)))


def hillshade_image(dem, lat_rows, water_mask_full=None):
    """A restrained, print-map style shaded relief as an RGB uint8 array."""
    px = _pixel_size_m(lat_rows)[:, None]
    z = cv2.GaussianBlur(dem, (0, 0), 0.8)
    gy, gx = np.gradient(z)
    gx /= px
    gy /= px
    slope = np.arctan(np.hypot(gx, gy) * 1.6)
    aspect = np.arctan2(-gx, gy)
    shade = np.zeros_like(dem)
    # Multi-directional light so north-facing Himalayan walls aren't black.
    for az, alt, w in ((315, 45, 0.55), (270, 50, 0.2), (0, 55, 0.25)):
        a, e = np.radians(az), np.radians(alt)
        shade += w * (np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect))
    shade = np.clip(shade, 0, 1)

    # Hypsometric tint: plains -> foothills -> high ground -> summits.
    stops = np.array([-50, 300, 1200, 2500, 3800, 5000, 6500, 9000], np.float32)
    cols = np.array([
        [204, 205, 190], [208, 207, 186], [196, 196, 170], [186, 180, 160],
        [194, 188, 176], [214, 212, 208], [238, 240, 242], [250, 251, 252],
    ], np.float32)
    rgb = np.stack([np.interp(dem, stops, cols[:, i]) for i in range(3)], -1)
    lit = (0.48 + 0.62 * shade)[..., None]
    out = np.clip(rgb * lit, 0, 255)
    if water_mask_full is not None:
        out[water_mask_full] = [150, 180, 196]
    return out.astype(np.uint8)


# --------------------------------------------------------------------------
# GIBS helpers
# --------------------------------------------------------------------------

def _colormap(name):
    """rgb -> numeric value, parsed from a GIBS colormap XML (cached)."""
    body = get(f"https://gibs.earthdata.nasa.gov/colormaps/v1.3/{name}.xml", cache_key=f"colormaps/{name}.xml")
    txt = body.decode("utf-8")
    out = {}
    for m in re.finditer(r'<ColorMapEntry rgb="(\d+),(\d+),(\d+)"[^>]*?sourceValue="\[?([0-9.,]+)', txt):
        r, g, b, v = m.groups()
        out[(int(r), int(g), int(b))] = float(v.split(",")[0])
    return out


def gibs_grid(layer, time, colormap, cache=True, nodata=np.nan):
    """Fetch a GIBS layer at GRID resolution over the frame and decode its
    colours back to data values. Transparent/unknown pixels -> nodata."""
    minx, miny, maxx, maxy = FRAME.merc_bbox()
    params = dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS=layer, STYLES="",
                  CRS="EPSG:3857", BBOX=f"{minx},{miny},{maxx},{maxy}",
                  WIDTH=FRAME.gw, HEIGHT=FRAME.gh, FORMAT="image/png", TRANSPARENT="TRUE", TIME=time)
    key = f"gibs/{layer}_{time}.png" if cache else None
    body = get(GIBS_WMS, params=params, cache_key=key, timeout=120)
    if body is None:
        return None
    a = np.asarray(Image.open(io.BytesIO(body)).convert("RGBA"))
    cmap = _colormap(colormap)
    packed = (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2]
    lut_keys = np.array([(r << 16) | (g << 8) | b for (r, g, b) in cmap], np.int32)
    lut_vals = np.array(list(cmap.values()), np.float32)
    order = np.argsort(lut_keys)
    lut_keys, lut_vals = lut_keys[order], lut_vals[order]
    idx = np.clip(np.searchsorted(lut_keys, packed), 0, len(lut_keys) - 1)
    hit = (lut_keys[idx] == packed) & (a[..., 3] > 0)
    out = np.full(packed.shape, nodata, np.float32)
    out[hit] = lut_vals[idx[hit]]
    return out


def land_cover():
    """IGBP class per GRID pixel (uint8, 0 = unknown). Latest annual map."""
    for year in range(dt.date.today().year, 2018, -1):
        g = gibs_grid("MODIS_Combined_L3_IGBP_Land_Cover_Type_Annual", f"{year}-01-01", "MODIS_IGBP_Land_Cover_Type", nodata=255)
        if g is not None and np.count_nonzero(g != 255) > g.size * 0.5:
            g[g == 0] = 17   # sourceValue "0,17" is water in this colormap
            g[g == 255] = 0  # no data
            return g.astype(np.uint8), year
    raise RuntimeError("No MODIS land cover year available from GIBS")


def recent_snow(days=8):
    """Max NDSI snow cover (0-100) over the last `days` Terra passes, and the
    date range used. Cloudy pixels simply contribute nothing."""
    acc = np.full((FRAME.gh, FRAME.gw), np.nan, np.float32)
    used = []
    today = dt.datetime.utcnow().date()
    for back in range(1, days + 3):
        d = today - dt.timedelta(days=back)
        g = gibs_grid("MODIS_Terra_NDSI_Snow_Cover", d.isoformat(), "MODIS_NDSI_Snow_Cover")
        if g is None:
            continue
        g[g > 100] = np.nan  # classification codes (cloud, night, ...)
        acc = np.fmax(acc, g)
        used.append(d)
        if len(used) >= days:
            break
    return np.nan_to_num(acc, nan=0.0), (min(used), max(used)) if used else (None, None)


def monthly_snow(n_years=3):
    """Average snow cover percent for each calendar month (index 0 = Jan),
    averaged over the `n_years` most recent years that have that month.
    Returns (12, gh, gw) uint8 and the (first, last) year used per month."""
    today = dt.date.today()
    out = np.zeros((12, FRAME.gh, FRAME.gw), np.uint8)
    years = []
    for m in range(1, 13):
        latest = today.year if m < today.month else today.year - 1
        acc, used = [], []
        for y in range(latest, latest - n_years - 2, -1):
            g = gibs_grid("MODIS_Terra_L3_Snow_Cover_Monthly_Average_Pct", f"{y}-{m:02d}-01", "MODIS_NDSI_Snow_Cover")
            if g is None or not np.isfinite(g).any():
                continue
            g[~np.isfinite(g) | (g > 100)] = 0  # transparent = snow-free
            acc.append(np.clip(g, 0, 100))
            used.append(y)
            if len(used) == n_years:
                break
        if acc:
            out[m - 1] = np.mean(acc, 0).astype(np.uint8)
            years.append([min(used), max(used)])
        else:
            years.append(None)
    return out, years


# --------------------------------------------------------------------------
# Derived lines
# --------------------------------------------------------------------------

def upper_edges(mask, dem, lc, min_elev, higher_by=40.0):
    """Pixels of `mask` that border a higher, non-mask, natural (not crop,
    urban or water) pixel -- the upper limit of that vegetation."""
    natural = ~np.isin(lc, [11, 12, 13, 14, 17, 0])
    edge = np.zeros_like(mask)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        nb_mask = np.roll(np.roll(mask, dy, 0), dx, 1)
        nb_dem = np.roll(np.roll(dem, dy, 0), dx, 1)
        nb_nat = np.roll(np.roll(natural, dy, 0), dx, 1)
        edge |= mask & ~nb_mask & nb_nat & (nb_dem > dem + higher_by)
    edge &= dem >= min_elev
    edge[:1, :] = edge[-1:, :] = False
    edge[:, :1] = edge[:, -1:] = False
    return edge


def arc_mask(lon_g, lat_g):
    """True on the Himalayan arc (the union of the sector bands)."""
    m = np.zeros(lon_g.shape, bool)
    for _, lo, hi, s, n in SECTORS:
        m |= _sector_mask(lon_g, lat_g, lo, hi, s, n)
    return m


def treeline(dem_g, lc, lon_g, lat_g):
    forest = np.isin(lc, list(FOREST))
    woody = forest | np.isin(lc, list(SHRUB))
    arc = arc_mask(lon_g, lat_g)
    tree_edge = upper_edges(forest, dem_g, lc, min_elev=2400) & arc
    shrub_edge = upper_edges(woody, dem_g, lc, min_elev=2800) & ~tree_edge & arc
    return tree_edge, shrub_edge


def _sector_mask(lon_g, lat_g, lo, hi, s, n):
    return (lon_g >= lo) & (lon_g < hi) & (lat_g >= s) & (lat_g < n)


def sector_stats(values_mask, dem_g, lon_g, lat_g, pct=50):
    out = []
    for name, lo, hi, s, n in SECTORS:
        sel = values_mask & _sector_mask(lon_g, lat_g, lo, hi, s, n)
        v = dem_g[sel]
        out.append({"sector": name, "lo": lo, "hi": hi,
                    "elev": int(np.percentile(v, pct)) if v.size > 30 else None,
                    "n": int(v.size)})
    return out


def seasonal_snowline(month_snow, dem_g, lon_g, lat_g, threshold=40):
    """For each sector and month: the elevation above which the ground is
    snow-covered at least `threshold` % of the time, from a smoothed
    100 m-bin profile. None when the profile never gets there."""
    bins = np.arange(1500, 7000, 100)
    res = []
    for name, lo, hi, s_, n_ in SECTORS:
        col = _sector_mask(lon_g, lat_g, lo, hi, s_, n_)
        row = []
        for m in range(12):
            s = month_snow[m].astype(np.float32)
            means = np.array([s[col & (dem_g >= b) & (dem_g < b + 100)].mean()
                              if (col & (dem_g >= b) & (dem_g < b + 100)).sum() > 40 else np.nan for b in bins])
            ok = np.isfinite(means)
            if ok.sum() < 5:
                row.append(None)
                continue
            prof = np.interp(bins, bins[ok], means[ok])
            prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
            above = np.where(prof >= threshold)[0]
            # Lowest elevation from which the profile stays above threshold.
            line = None
            for i in above:
                if (prof[i:] >= threshold - 10).all():
                    line = int(bins[i])
                    break
            row.append(line)
        res.append({"sector": name, "months": row})
    return res
