"""Regional cultural atlases: temples, forts, peaks, protected areas,
forests, birds, GI products and visitor numbers, plus places to stay and
drive-time circuits for planning your own tour. One page per region (see
REGIONS at the end of this file); Kumaon is the first.

A slow-refresh page: its sources change over months, not hours, so it's
rebuilt monthly by .github/workflows/culture.yml (or locally: python -m
nowcast.culture [--region kumaon]) and published to the culture branch,
which the Pages build copies over static/<region>/culture/. That keeps
Overpass, Wikidata and eBird out of the two-hourly run.

Places to stay can't be rated on hospitality or food hygiene from open data,
and review sites forbid scraping, so stays are described by facts that can
be checked: run by the state (the region's tourism corporation's rest
houses, such as KMVN's), the road that reaches them, food on site or within
walking distance, and what OSM records about them. Only business names and
localities are shown - no phone numbers or personal names.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np

from .geo import CACHE, get, get_json, make_frame

OVERPASS = "https://overpass-api.de/api/interpreter"
CDIR = os.path.join(CACHE, "culture")
os.makedirs(CDIR, exist_ok=True)


@dataclass(frozen=True)
class Corp:
    """A state tourism corporation's rest houses: where to fetch them, and
    where each one is (their pages have no coordinates)."""
    abbr: str                 # "KMVN"
    name: str                 # "Kumaon Mandal Vikas Nigam"
    site: str                 # "kmvn.in"
    fetch: Callable[[], list]  # -> [{slug, name, address, blurb, rooms}]
    url: Callable[[str], str]  # slug -> the rest house's page
    place: dict               # slug -> OSM place names (with alternative spellings)
    fallback: dict = dc_field(default_factory=dict)  # slug -> approximate (lat, lon)
    district: dict = dc_field(default_factory=dict)  # slug -> district, where a place name recurs
    # When the corporation's own site can't be read, its rest houses are the
    # OSM lodgings whose name or operator matches this (none are fetched).
    osm_match: re.Pattern | None = None
    cache_glob: str = ""      # its fetched pages under cache/, for the page's as-of date


@dataclass(frozen=True)
class Region:
    """Everything that differs between one region's cultural atlas and
    another's. Fetchers and the page itself are shared."""
    id: str                   # "kumaon": output under static/<id>/culture/
    name: str                 # "Kumaon"
    box: tuple                # w, s, e, n
    districts: list           # OSM admin_level 5 names, in display order
    ebird: dict               # eBird subnational2 code -> district
    towns: Callable[[], list]  # -> [(name, lat, lon, district)], always routable destinations
    corp: Corp
    gi_words: re.Pattern      # GI products named for a place in the region
    utdb: dict                # UTDB entity -> (destination name, district)
    circuits: list            # (title, blurb, [overnight bases])
    shade_alt: str            # alt text for the relief image
    gi_note: str              # HTML: crafts registered state-wide but practised here
    treks: str                # a few treks on the map, for the page's limits note
    pa_skip: frozenset = frozenset()       # protected areas across the state line
    wd_bad_coords: frozenset = frozenset()  # Wikidata items placed well off
    cache_dir: str = ""       # under cache/culture/ (Kumaon's predates regions: "")
    store: str = ""           # localStorage key prefix; the pages share an origin
    back: tuple = ()          # (href, label) links above the title
    nowcast: str = ""         # relative link to the region's weather nowcast, if any

    @property
    def frame(self):
        w, s, e, n = self.box
        return make_frame(w, e, s, n, z=10)

    def ckey(self, name):
        """Cache key for a region-specific fetch."""
        return "/".join(p for p in ("culture", self.cache_dir, name) if p)


# How long each source's cached copy is used before it's fetched again. The
# workflow rebuilds monthly, so each expires a few days short of its cycle;
# a source that's down when its copy has expired falls back to that copy.
DAY = 86400
MONTHLY, BIMONTHLY, QUARTERLY, YEARLY = 25 * DAY, 55 * DAY, 85 * DAY, 360 * DAY


def _expired(key, max_age):
    p = os.path.join(CACHE, key)
    return not os.path.exists(p) or time.time() - os.path.getmtime(p) >= max_age


def _overpass(R, q, key, max_age):
    body = get(OVERPASS, params={"data": q}, timeout=400, cache_key=R.ckey(key), max_age=max_age, stale_ok=True)
    return json.loads(body)["elements"]


def _bbox_ql(R):
    w, s, e, n = R.box
    return f"({s},{w},{n},{e})"


# --------------------------------------------------------------------------
# Fetchers (each cached under cache/culture/ until it expires; delete a
# file to refresh it sooner)
# --------------------------------------------------------------------------

def osm_pois(R):
    b = _bbox_ql(R)
    q = (f'[out:json][timeout:300];('
         f'nwr["tourism"~"^(hotel|guest_house|hostel|chalet|alpine_hut|camp_site|apartment|motel|attraction|'
         f'viewpoint|museum|artwork)$"]{b};'
         f'nwr["amenity"~"^(place_of_worship|restaurant|cafe|fast_food)$"]{b};'
         f'nwr["historic"]{b};nwr["natural"~"^(peak|cave_entrance|waterfall)$"]["name"]{b};'
         f'nwr["boundary"~"^(protected_area|national_park)$"]{b};nwr["leisure"="nature_reserve"]{b};'
         f');out center tags qt;')
    return _overpass(R, q, "osm_poi.json", MONTHLY)


def osm_roads(R):
    q = (f'[out:json][timeout:380];way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|'
         f'trunk_link|primary_link|secondary_link|tertiary_link)$"]{_bbox_ql(R)};out body geom qt;')
    return _overpass(R, q, "osm_roads.json", BIMONTHLY)


def osm_protected(R):
    b = _bbox_ql(R)
    q = (f'[out:json][timeout:300];(relation["boundary"~"^(protected_area|national_park)$"]{b};'
         f'way["boundary"~"^(protected_area|national_park)$"]{b};'
         f'relation["leisure"="nature_reserve"]{b};way["leisure"="nature_reserve"]{b};);'
         f'out body geom qt;')
    return _overpass(R, q, "osm_protected.json", BIMONTHLY)


def osm_places(R):
    q = f'[out:json][timeout:300];node["place"~"^(city|town|village|hamlet|locality)$"]["name"]{_bbox_ql(R)};out qt;'
    return _overpass(R, q, "osm_places.json", MONTHLY)


def _page_text(html):
    t = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", html, flags=re.S)
    t = re.sub(r"<[^>]+>", "\n", t).replace("&nbsp;", " ").replace("&amp;", "&")
    return [ln.strip() for ln in t.splitlines() if ln.strip()]


def kmvn_houses():
    """KMVN's tourist rest houses: name, address line, blurb and room types,
    from kmvn.in (robots.txt allows it; fetched once, a second apart)."""
    listing = get("https://kmvn.in/hotels", timeout=60, cache_key="culture/kmvn/hotels.html", max_age=BIMONTHLY,
                  stale_ok=True).decode("utf-8", "ignore")
    slugs = sorted(set(re.findall(r"/hotels/([a-z0-9\-]+)", listing)) - {"test-hotel"})
    out = []
    for slug in slugs:
        key = f"culture/kmvn/{slug}.html"
        fresh = _expired(key, BIMONTHLY)
        html = get(f"https://kmvn.in/hotels/{slug}", timeout=60, cache_key=key, max_age=BIMONTHLY, stale_ok=True)
        if fresh:
            time.sleep(1.0)
        if not html:
            continue
        lines = _page_text(html.decode("utf-8", "ignore"))
        title = re.sub(r"\s+-\s+Hotels$", "", lines[0]).strip()
        title = re.sub(r"\s+", " ", title)
        addr = next((ln for ln in lines if re.search(r",\s*Uttarakhand", ln, re.I)), "")
        try:
            i = lines.index("Book now")
            j = lines.index("Let us know what you think")
            blurb = " ".join(lines[i + 1:j])
        except ValueError:
            blurb = ""
        try:
            r0 = lines.index("rooms") + 1
            r1 = lines.index("Kumaon Experience")
            rooms = sorted({ln for ln in lines[r0:r1] if re.match(r"^[A-Za-z][A-Za-z ()&\-]+$", ln)
                            and not ln.startswith("Capacity")})
        except ValueError:
            rooms = []
        out.append({"slug": slug, "name": title, "address": addr, "blurb": re.sub(r"\s+", " ", blurb).strip(),
                    "rooms": rooms})
    return out


EBIRD = "https://api.ebird.org/v2"


def _ebird_key():
    """EBIRD_API_KEY, else .secrets/ebird_key in the repo (gitignored) or
    beside it."""
    k = os.environ.get("EBIRD_API_KEY")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in (repo, os.path.dirname(repo)):
        p = os.path.join(d, ".secrets", "ebird_key")
        if not k and os.path.exists(p):
            k = open(p, encoding="utf-8").read().strip()
    return k


def _ebird(path, key, params=None, max_age=MONTHLY):
    import requests
    p = os.path.join(CDIR, "ebird", key)
    if not _expired(f"culture/ebird/{key}", max_age):
        return json.load(open(p, encoding="utf-8"))
    token = _ebird_key()
    try:
        if not token:
            raise RuntimeError("no eBird API key: export EBIRD_API_KEY (a shell variable set without export "
                               "isn't passed to Python) or put it in .secrets/ebird_key")
        r = requests.get(f"{EBIRD}/{path}", params=params or {}, headers={"X-eBirdApiToken": token}, timeout=120)
        r.raise_for_status()
    except (RuntimeError, requests.RequestException) as e:
        if os.path.exists(p):
            print(f"  eBird {path} failed, using the cached copy: {e}", flush=True)
            return json.load(open(p, encoding="utf-8"))
        raise
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(r.text)
    time.sleep(0.5)
    return r.json()


def ebird(R):
    """Hotspots and species lists per district, with English and
    scientific names from eBird's taxonomy."""
    tax = {t["speciesCode"]: t for t in _ebird("ref/taxonomy/ebird", "taxonomy.json", {"fmt": "json"}, YEARLY)}
    out = {"districts": [], "hotspots": []}
    all_species = set()
    for code, name in R.ebird.items():
        spp = [s for s in _ebird(f"product/spplist/{code}", f"spplist_{code}.json")
               if tax.get(s, {}).get("category") == "species"]
        all_species.update(spp)
        hs = _ebird(f"ref/hotspot/{code}", f"hotspots_{code}.json", {"fmt": "json"})
        out["districts"].append({"code": code, "district": name, "species": len(spp), "hotspots": len(hs)})
        for h in hs:
            out["hotspots"].append({"id": h["locId"], "name": h["locName"], "lat": h["lat"], "lon": h["lng"],
                                    "species": h.get("numSpeciesAllTime") or 0, "district": name,
                                    "last": h.get("latestObsDt")})
    out["species_total"] = len(all_species)
    out["taxonomy"] = tax
    out["species_codes"] = all_species
    return out


WORLDCOVER = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif")
WC_RES = 0.001  # degrees (~100 m): read from the COG overviews, not the 10 m data


def _wc_tiles(box):
    """South-west corners of the 3-degree WorldCover tiles covering box."""
    w, s, e, n = box
    return [(tlat, tlon) for tlat in range(math.floor(s / 3) * 3, math.ceil(n), 3)
            for tlon in range(math.floor(w / 3) * 3, math.ceil(e), 3)]


def worldcover(R):
    """ESA WorldCover 2021 classes on a 0.001-degree lat/lon grid over the
    region's box (north-up; row 0 = north edge). Read straight from the
    cloud-optimised GeoTIFFs' overviews (needs rasterio, a local-only
    dependency)."""
    p = os.path.join(CACHE, R.ckey("worldcover.npz"))
    if os.path.exists(p):
        z = np.load(p)
        return z["cls"]
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    w, s, e, n = R.box
    W, H = int(round((e - w) / WC_RES)), int(round((n - s) / WC_RES))
    out = np.zeros((H, W), np.uint8)
    for tlat, tlon in _wc_tiles(R.box):
        bw, bs, be, bn = max(w, tlon), max(s, tlat), min(e, tlon + 3), min(n, tlat + 3)
        if be <= bw or bn <= bs:
            continue
        with rasterio.open(WORLDCOVER.format(lat=tlat, lon=tlon)) as d:
            win = from_bounds(bw, bs, be, bn, d.transform)
            shape = (int(round((bn - bs) / WC_RES)), int(round((be - bw) / WC_RES)))
            a = d.read(1, window=win, out_shape=shape, resampling=Resampling.nearest)
        r0, c0 = int(round((n - bn) / WC_RES)), int(round((bw - w) / WC_RES))
        out[r0:r0 + shape[0], c0:c0 + shape[1]] = a
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, cls=out)
    return out


GIR = "https://search.ipindia.gov.in/GIRPublicSearch/"
# Uttarakhand applications in the GI registry (application numbers), found by
# searching its list for Uttarakhand products and places. Each is checked
# against its registry page at build time (status, goods, area).
GI_APPS = [415, 419, 520, 589, 648, 649, 650, 651, 652, 653, 654, 829, 832, 835, 841, 844, 852, 856, 860, 863,
           864, 866, 867, 873, 874, 905, 907, 919, 1082, 1188, 1189, 1206, 1390, 1703, 1704, 1705, 1706, 1850,
           2069, 2070, 2071]
GI_LABELS = {"application number", "geographical indications", "status", "applicant name", "applicant address",
             "date of filing", "class", "goods", "geographical area", "priority country", "journal number",
             "availability date", "certificate number", "certificate date", "registration valid upto"}


def gi_products():
    """Uttarakhand's GI applications, shared by every region (build() marks
    the ones named for a place in the region). Application-level fields
    only: the registry pages also list every authorised user's name and
    address; those are never kept."""
    out = []
    for app in GI_APPS:
        key = f"culture/gi/{app}.html"
        fresh = _expired(key, QUARTERLY)
        html = get(f"{GIR}Application/Details/{app}", timeout=90, cache_key=key, max_age=QUARTERLY, stale_ok=True)
        if fresh:
            time.sleep(1.0)
        if not html:
            continue
        lines = _page_text(html.decode("utf-8", "ignore"))

        def field(label):
            try:
                i = next(k for k, ln in enumerate(lines) if ln.rstrip(": ").strip().lower() == label.lower())
            except StopIteration:
                return ""
            v = lines[i + 1].strip()
            return "" if v.endswith(":") or v.lower() in GI_LABELS else v

        rec = {"app": app, "name": field("Geographical Indications"), "status": field("Status"),
               "goods": field("Goods"), "class": field("Class"), "area": field("Geographical Area"),
               "filed": field("Date of Filing"), "certificate_date": field("Certificate Date"),
               "url": f"{GIR}Application/Details/{app}"}
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# Region outline
# --------------------------------------------------------------------------

def districts_shapes(R):
    """{district: shapely geometry} from OSM's district boundaries
    (admin_level 5), assembled from their member ways."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    q = (f'[out:json][timeout:300];relation["boundary"="administrative"]["admin_level"="5"]{_bbox_ql(R)};'
         f'out body geom;')
    out = {}
    for r in _overpass(R, q, "osm_districts.json", QUARTERLY):
        name = re.sub(r"\s+district$", "", r.get("tags", {}).get("name:en") or r.get("tags", {}).get("name") or "",
                      flags=re.I)
        if name not in R.districts:
            continue
        lines = [LineString([(g["lon"], g["lat"]) for g in m["geometry"]]) for m in r.get("members", [])
                 if m.get("type") == "way" and m.get("role") in ("outer", "") and m.get("geometry")]
        polys = list(polygonize(unary_union(lines)))
        if polys:
            out[name] = unary_union(polys).buffer(0)
    missing = set(R.districts) - set(out)
    if missing:
        raise RuntimeError(f"OSM district boundaries missing: {missing}")
    return out


class Locator:
    def __init__(self, R):
        from shapely.ops import unary_union
        from shapely.prepared import prep
        self.shapes = districts_shapes(R)
        self.prepared = {k: prep(v) for k, v in self.shapes.items()}
        self.union = unary_union(list(self.shapes.values()))
        self.near = prep(self.union.buffer(0.03))  # ~3 km: roads and stays just over a line

    def district(self, lon, lat):
        from shapely.geometry import Point
        pt = Point(lon, lat)
        for k, p in self.prepared.items():
            if p.contains(pt):
                return k
        return None

    def inside(self, lon, lat, pad=False):
        from shapely.geometry import Point
        return self.near.contains(Point(lon, lat)) if pad else self.district(lon, lat) is not None


# --------------------------------------------------------------------------
# Roads: OSM ways split at junctions into a routable graph
# --------------------------------------------------------------------------

ROAD_CLASS = {"motorway": 0, "trunk": 0, "trunk_link": 0, "primary": 1, "primary_link": 1,
              "secondary": 2, "secondary_link": 2, "tertiary": 3, "tertiary_link": 3, "unclassified": 4}
ROAD_NAMES = ["National highway", "Primary road", "Secondary road", "Tertiary road", "Minor road"]
# Average driving speeds (km/h), rough figures for winding hill roads and the
# flatter bhabar and terai. They give Haldwani-Almora about 3 hours and
# Bhimtal-Nainital about 45 minutes; treat every time as an estimate.
SPEED_HILL = [32, 30, 25, 20, 15]
SPEED_PLAINS = [55, 48, 38, 30, 22]
PLAINS_M = 450  # below this, the bhabar and terai: straighter, flatter roads
# Broad valley floors above PLAINS_M drive like the plains too: the Doon
# valley sits at 450-700 m. Roads count as flat where the ground within
# FLAT_KM averages under FLAT_SLOPE degrees; valley floors and the bhabar
# average 1-3, hill towns and gorges 11-22.
FLAT_SLOPE = 4.0
FLAT_KM = 1.5


def _hav_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def road_graph(R, loc, elev_at, flat_at=lambda lon, lat: False):
    """Edges between junctions: {a, b, cls, km, min, geom[(lon, lat)...]}.
    Keeps ways touching the region (with a 3 km margin) and then the largest
    connected network, so isolated fragments can't trap a route."""
    import networkx as nx
    ways = []
    for w in osm_roads(R):
        cls = ROAD_CLASS.get(w.get("tags", {}).get("highway"))
        geom = w.get("geometry")
        if cls is None or not geom or len(geom) != len(w["nodes"]):
            continue
        if not any(loc.inside(g["lon"], g["lat"], pad=True) for g in geom[::max(1, len(geom) // 6)] + [geom[-1]]):
            continue
        ways.append((cls, w["nodes"], [(g["lon"], g["lat"]) for g in geom], w.get("tags", {})))
    uses = {}
    for _, nodes, _, _ in ways:
        for k, n in enumerate(nodes):
            uses[n] = uses.get(n, 0) + (2 if k in (0, len(nodes) - 1) else 1)
    edges = []
    for cls, nodes, pts, tags in ways:
        start = 0
        for k in range(1, len(nodes)):
            if uses[nodes[k]] > 1 or k == len(nodes) - 1:
                seg = pts[start:k + 1]
                if nodes[start] != nodes[k] and len(seg) >= 2:
                    a = np.array(seg)
                    km = float(_hav_km(a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1]).sum())
                    mid = seg[len(seg) // 2]
                    sp = (SPEED_PLAINS if elev_at(*mid) < PLAINS_M or flat_at(*mid) else SPEED_HILL)[cls]
                    edges.append({"a": nodes[start], "b": nodes[k], "cls": cls, "km": km, "min": km / sp * 60,
                                  "geom": seg, "ref": tags.get("ref") or ""})
                start = k
    G = nx.Graph()
    for i, e in enumerate(edges):
        if G.has_edge(e["a"], e["b"]) and G[e["a"]][e["b"]]["min"] <= e["min"]:
            continue
        G.add_edge(e["a"], e["b"], min=e["min"], i=i)
    big = max(nx.connected_components(G), key=len)
    keep = {d["i"] for a, b, d in G.edges(data=True) if a in big}
    return _merge_chains([e for i, e in enumerate(edges) if i in keep])


def _merge_chains(edges):
    """Join edges through nodes where exactly two edges of the same class
    meet (way boundaries that aren't junctions)."""
    inc = {}
    for i, e in enumerate(edges):
        inc.setdefault(e["a"], []).append(i)
        inc.setdefault(e["b"], []).append(i)
    alive = [True] * len(edges)
    for n, ids in inc.items():
        ids = [i for i in ids if alive[i]]
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        e1, e2 = edges[ids[0]], edges[ids[1]]
        if e1["cls"] != e2["cls"]:
            continue
        g1 = e1["geom"] if e1["b"] == n else e1["geom"][::-1]
        a = e1["a"] if e1["b"] == n else e1["b"]
        g2 = e2["geom"] if e2["a"] == n else e2["geom"][::-1]
        b = e2["b"] if e2["a"] == n else e2["a"]
        if a == b:
            continue
        merged = {"a": a, "b": b, "cls": e1["cls"], "km": e1["km"] + e2["km"], "min": e1["min"] + e2["min"],
                  "geom": g1 + g2[1:], "ref": e1["ref"] or e2["ref"]}
        edges[ids[0]] = merged
        alive[ids[1]] = False
        for end in (a, b):
            lst = inc[end]
            inc[end] = [ids[0] if i == ids[1] else i for i in lst]
    return [e for e, ok in zip(edges, alive) if ok]


def wikidata_sites(R):
    w, s, e, n = R.box
    q = f"""SELECT ?item ?itemLabel ?coord ?type ?typeLabel ?article ?image ?links WHERE {{
  SERVICE wikibase:box {{ ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:cornerSouthWest "Point({w} {s})"^^geo:wktLiteral .
    bd:serviceParam wikibase:cornerNorthEast "Point({e} {n})"^^geo:wktLiteral . }}
  ?item wikibase:sitelinks ?links .
  OPTIONAL {{ ?item wdt:P31 ?type . }}
  OPTIONAL {{ ?item wdt:P18 ?image . }}
  OPTIONAL {{ ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,hi" . }}
}}"""
    body = get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}, timeout=300,
               cache_key=R.ckey("wikidata.json"), max_age=MONTHLY, stale_ok=True)
    return json.loads(body)["results"]["bindings"]


GBIF_TILE = "https://api.gbif.org/v2/map/occurrence/density/{z}/{x}/{y}@1x.png"


def gbif_birds(frame):
    """GBIF occurrence density for birds (class Aves, taxonKey 212) on the
    frame's GRID: GBIF's 512-pixel tiles one zoom level below GRID's line up
    with it exactly. Each coloured pixel is a cell with records; the colour
    steps with the count."""
    import io
    from PIL import Image
    z = frame.z - 2  # GRID is z-1 at 256 px; 512 px tiles at z-2 match
    gx0, gy0 = frame.x0 // 2, frame.y0 // 2
    tx0, ty0 = gx0 // 512, gy0 // 512
    tx1, ty1 = (gx0 + frame.gw - 1) // 512, (gy0 + frame.gh - 1) // 512
    canvas = Image.new("RGBA", ((tx1 - tx0 + 1) * 512, (ty1 - ty0 + 1) * 512))
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            body = get(GBIF_TILE.format(z=z, x=tx, y=ty), params={"taxonKey": 212, "style": "classic.point"},
                       timeout=60, cache_key=f"culture/gbif/aves_{z}_{tx}_{ty}.png", max_age=MONTHLY, stale_ok=True)
            if body:
                canvas.paste(Image.open(io.BytesIO(body)).convert("RGBA"), ((tx - tx0) * 512, (ty - ty0) * 512))
    ox, oy = gx0 - tx0 * 512, gy0 - ty0 * 512
    return np.asarray(canvas.crop((ox, oy, ox + frame.gw, oy + frame.gh)))


# --------------------------------------------------------------------------
# Sites: OSM and Wikidata, merged
# --------------------------------------------------------------------------

WD_KIND = [
    ("temple", r"temple|gurdwara|mosque|monastery|church|shrine|ashram|shaktipeeth|dargah|math\b"),
    ("heritage", r"fort|archaeolog|monument|museum|palace|ruin|castle|heritage|memorial|stepwell|naula|"
                 r"historic|rock art|cemetery|observatory"),
    ("nature", r"lake|waterfall|glacier|mountain pass|valley|cave|hot spring|protected area|national park|"
               r"sanctuary|reserve|viewpoint|botanical|zoo|meadow|bugyal"),
]
OSM_KIND = {
    "amenity=place_of_worship": "temple", "historic=wayside_shrine": "temple",
    "tourism=museum": "heritage", "tourism=viewpoint": "nature", "natural=cave_entrance": "nature",
    "natural=waterfall": "nature", "tourism=attraction": "attraction",
}
KIND_LABEL = {"temple": "Temples and shrines", "heritage": "Forts, ruins and museums",
              "nature": "Viewpoints, lakes, caves, glaciers", "attraction": "Other attractions", "peak": "Peaks"}


def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"\b(temple|mandir|shri|sri|shree|devi|dham|the|of|ji|mandi)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _osm_kind(t):
    for k in ("amenity", "tourism", "natural"):
        if f"{k}={t.get(k)}" in OSM_KIND:
            return OSM_KIND[f"{k}={t.get(k)}"]
    if "historic" in t and t["historic"] not in ("boundary_stone", "yes"):
        return "heritage"
    return None


def _latin(s):
    """Name without diacritics, or None if it isn't in Latin script."""
    import unicodedata
    if not s:
        return None
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    if not re.match(r"^[A-Za-z]", s):
        return None
    return s[:1].upper() + s[1:]


def _commons_thumb(url, width=360):
    name = url.rsplit("/", 1)[-1]
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{name}?width={width}"


def _num(s):
    try:
        return float(str(s).lower().replace("m", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def sites(R, loc):
    from scipy.spatial import cKDTree
    wd = {}
    for r in wikidata_sites(R):
        q = r["item"]["value"].rsplit("/", 1)[1]
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r["coord"]["value"])
        if not m:
            continue
        it = wd.setdefault(q, {"qid": q, "name": r["itemLabel"]["value"], "lon": float(m.group(1)),
                               "lat": float(m.group(2)), "links": int(r["links"]["value"]), "types": set(),
                               "wiki": r.get("article", {}).get("value"), "image": None})
        if "typeLabel" in r:
            it["types"].add(r["typeLabel"]["value"])
        if "image" in r and not it["image"]:
            it["image"] = _commons_thumb(r["image"]["value"])
    for it in wd.values():
        it["kind"] = None
        if re.match(r"^Q\d+$", it["name"]):
            continue
        tl = " | ".join(sorted(it["types"])).lower()
        for kind, pat in WD_KIND:
            if re.search(pat, tl) and not re.search(r"bank|gas station|hotel|school|railway", tl):
                it["kind"] = kind
                break
        if it["kind"] is None and re.search(r"\bmountain\b|\bhill\b|\bpeak\b", tl) and "hill station" not in tl:
            it["kind"] = "peak"

    for q in [q for q, it in wd.items() if it["name"] in R.wd_bad_coords]:
        wd.pop(q)
    out = []
    used_wd = set()
    for e in osm_pois(R):
        t = e.get("tags", {})
        name = t.get("name:en") or t.get("name")
        lat, lon = (e.get("lat"), e.get("lon")) if "lat" in e else (e["center"]["lat"], e["center"]["lon"])
        if not name:
            continue
        kind = "peak" if t.get("natural") == "peak" else _osm_kind(t)
        if not kind:
            continue
        dist = loc.district(lon, lat)
        if not dist:
            continue
        rec = {"name": name, "lon": lon, "lat": lat, "kind": kind, "district": dist,
               "osm": f"{e['type']}/{e['id']}", "religion": t.get("religion"),
               "ele": _num(t.get("ele")), "links": 0, "wiki": None, "image": None, "qid": t.get("wikidata")}
        if kind == "peak" and rec["ele"] is None:
            continue
        if rec["qid"] in wd:
            _attach(rec, wd[rec["qid"]])
            used_wd.add(rec["qid"])
        out.append(rec)
    # Wikidata items that OSM didn't link: merge by name within 1.5 km, else add.
    tree = cKDTree([(r["lon"] * 0.87, r["lat"]) for r in out])
    for q, it in wd.items():
        if q in used_wd or not it["kind"]:
            continue
        dist = loc.district(it["lon"], it["lat"])
        if not dist:
            continue
        # Same name within 1.5 km; or within 40 km if only one such name, as
        # some Wikidata coordinates are well off (Patal Bhuvaneshwar's is 50 km).
        key = _norm(it["name"])
        near = [i for i in tree.query_ball_point((it["lon"] * 0.87, it["lat"]), 0.015) if _norm(out[i]["name"]) == key]
        wide = [i for i in tree.query_ball_point((it["lon"] * 0.87, it["lat"]), 0.6) if _norm(out[i]["name"]) == key]
        match = out[near[0]] if key and near else out[wide[0]] if key and len(wide) == 1 else None
        if match:
            _attach(match, it)
        else:
            rec = {"name": it["name"], "lon": it["lon"], "lat": it["lat"], "kind": it["kind"], "district": dist,
                   "osm": None, "religion": None, "ele": None, "links": 0, "wiki": None, "image": None, "qid": q}
            _attach(rec, it)
            out.append(rec)
    for r in out:
        r["score"] = r["links"] + (3 if r["wiki"] else 0) + (1 if r["image"] else 0)
    return out


def _attach(rec, it):
    rec["qid"] = it["qid"]
    rec["links"] = max(rec["links"], it["links"])
    rec["wiki"] = rec["wiki"] or it["wiki"]
    rec["image"] = rec["image"] or it["image"]
    if re.match(r"^[A-Za-z]", it["name"]) and not re.match(r"^[A-Za-z]", rec["name"]):
        rec["name"] = it["name"]


# --------------------------------------------------------------------------
# Places to stay
# --------------------------------------------------------------------------

STAY_TYPES = {"hotel": "Hotel", "guest_house": "Guest house / homestay", "hostel": "Hostel", "chalet": "Cottage",
              "alpine_hut": "Mountain hut", "camp_site": "Camp", "apartment": "Apartment", "motel": "Motel"}
FOOD_AMENITIES = {"restaurant", "cafe", "fast_food"}
# Where each KMVN rest house is, as an OSM place name (with the alternative
# spellings OSM uses) - KMVN's pages have no coordinates.
KMVN_PLACE = {
    "gyan-vriksh-trh-kakrighat": ["Kakrighat"], "holiday-home-dhikuli": ["Dhikuli", "Dhikuli Village"],
    "jungle-camp-sigri-tent-sharing-accommodation": ["Sigri", "Pangot"],
    "nine-corner-retreat-trh-naukuchiyatal": ["Naukuchiatal", "Naukuchiyatal", "Naukuchia Tal"],
    "sarovar-bliss-trh-bhimtal": ["Bhimtal"], "trh-abbott-mount": ["Abbott Mount", "Abott Mount"],
    "trh-almora": ["Almora"], "trh-baijnath": ["Baijnath"], "trh-bhikiyasen": ["Bhikiyasain", "Bhikiyasen"],
    "trh-bhowali": ["Bhowali"], "trh-binsar": ["Binsar"], "trh-birthi": ["Birthi"], "trh-champawat": ["Champawat"],
    "trh-chaukori": ["Chaukori", "Chokori"], "trh-chilianaula-ranikhet-himadri": ["Chilianaula", "Ranikhet"],
    "trh-deenapani": ["Dinapani", "Deenapani"], "trh-dharchula": ["Dharchula"], "trh-didihat": ["Didihat"],
    "trh-dudhauli": ["Dudholi", "Dudhauli", "Dudholi Dhar"], "trh-gangolihat": ["Gangolihat"],
    "trh-jageshwar": ["Jageshwar"], "trh-jaspur": ["Jaspur"], "trh-katarmal": ["Katarmal"],
    "trh-kausani-trishul": ["Kausani"], "trh-khairna": ["Khairna"], "trh-lohaghat": ["Lohaghat"],
    "trh-mohaan": ["Mohan"], "trh-monal-ranikhet": ["Ranikhet"], "trh-mukteshwar": ["Mukteshwar"],
    "trh-munsyari": ["Munsiyari", "Munsyari", "Munsiari"], "trh-nanakmatta": ["Nanakmatta"],
    "trh-narayan-ashram": ["Narayan Ashram"], "trh-padampuri": ["Padampuri"],
    "trh-parichay-naukuchiyatal": ["Naukuchiatal", "Naukuchiyatal", "Naukuchia Tal"],
    "trh-patal-bhuvneshwar": ["Patal Bhuvaneshwar", "Patal Bhubaneshwar", "Bhubaneshwar"],
    "trh-pithoragarh": ["Pithoragarh"], "trh-ramgarh": ["Ramgarh"], "trh-ramnagar": ["Ramnagar"],
    "trh-ranikhet-kalika": ["Kalika", "Ranikhet"], "trh-sattal": ["Sattal", "Sat Tal"],
    "trh-snow-view-nainital-snow-view-heritage": ["Nainital"], "trh-sukhatal-nainital-mount-view": ["Nainital"],
    "trh-tallital-nainital-sarovar": ["Nainital"], "trh-tanakpur": ["Tanakpur"],
    "trh-trh-sitlakhet": ["Sitlakhet", "Shitlakhet", "Shitalakhet"],
}
# Approximate positions (lat, lon) for rest houses whose place OSM doesn't
# name; shown as approximate on the page.
KMVN_FALLBACK = {"trh-narayan-ashram": (30.032, 80.585), "trh-binsar": (29.705, 79.756),
                 "trh-patal-bhuvneshwar": (29.689, 80.103), "jungle-camp-sigri-tent-sharing-accommodation": (29.42, 79.41),
                 "holiday-home-dhikuli": (29.470, 79.135), "nine-corner-retreat-trh-naukuchiyatal": (29.326, 79.588),
                 "trh-parichay-naukuchiyatal": (29.322, 79.585), "trh-abbott-mount": (29.383, 80.063),
                 "trh-kausani-trishul": (29.843, 79.601), "trh-dudhauli": (29.818, 79.337), "trh-nanakmatta": (28.944, 79.824), "trh-padampuri": (29.385, 79.655)}
# Where a place name recurs in Kumaon, the district the rest house is in.
KMVN_DISTRICT = {"trh-kausani-trishul": "Bageshwar", "trh-ramgarh": "Nainital", "trh-ranikhet-kalika": "Almora",
                 "trh-monal-ranikhet": "Almora", "trh-katarmal": "Almora", "trh-jageshwar": "Almora",
                 "trh-baijnath": "Bageshwar", "trh-champawat": "Champawat", "trh-lohaghat": "Champawat"}
MEAL_PLANS = {"EP": "room only", "CP": "breakfast included", "MAP": "breakfast and dinner included"}


def _meal_plan(rooms):
    plans = set()
    for r in rooms:
        u = r.upper()
        for p in ("MAP", "CP", "EP"):
            if re.search(rf"\b{p}\b", u) or (p == "CP" and "C P" in u):
                plans.add(p)
                break
    return "MAP" if "MAP" in plans else "CP" if "CP" in plans else "EP" if "EP" in plans else None


class Places:
    """Settlement names (OSM place nodes) for locality labels and rest-house placement."""

    def __init__(self, R, loc):
        from scipy.spatial import cKDTree
        rank = {"city": 0, "town": 1, "village": 2, "hamlet": 3, "locality": 4}
        self.items = []
        for e in osm_places(R):
            t = e.get("tags", {})
            name = _latin(t.get("name:en")) or _latin(t.get("name"))
            if name and loc.inside(e["lon"], e["lat"], pad=True):
                self.items.append({"name": name, "alt": {_norm(t.get(k)) for k in ("name", "name:en", "alt_name",
                                                                                    "old_name") if t.get(k)},
                                   "lon": e["lon"], "lat": e["lat"], "rank": rank.get(t.get("place"), 5)})
        self.tree = cKDTree([(p["lon"] * 0.87, p["lat"]) for p in self.items])
        self.towns = [p for p in self.items if p["rank"] <= 2]
        self.ttree = cKDTree([(p["lon"] * 0.87, p["lat"]) for p in self.towns])

    def locality(self, lon, lat, towns_only=False):
        tree, items = (self.ttree, self.towns) if towns_only else (self.tree, self.items)
        d, i = tree.query((lon * 0.87, lat))
        return items[i]["name"], round(float(d) * 111, 1)

    def find(self, names, loc=None, district=None):
        for k in (_norm(n) for n in names):
            hits = [p for p in self.items if k in p["alt"]]
            if district:
                hits = [p for p in hits if loc.district(p["lon"], p["lat"]) == district]
            if hits:
                return min(hits, key=lambda p: p["rank"])
        return None


def stays(R, loc, places, road_tree, road_cls, houses):
    """Lodging from OSM and the state tourism corporation, with checkable
    facts only (no phone numbers, emails or operators are read)."""
    from scipy.spatial import cKDTree
    C = R.corp
    food, lodging = [], []
    for e in osm_pois(R):
        t = e.get("tags", {})
        lat, lon = (e.get("lat"), e.get("lon")) if "lat" in e else (e["center"]["lat"], e["center"]["lon"])
        if t.get("amenity") in FOOD_AMENITIES:
            food.append((lon * 0.87, lat))
        if t.get("tourism") in STAY_TYPES and (t.get("name:en") or t.get("name")) and loc.district(lon, lat):
            lodging.append((e, t, lon, lat))
    ftree = cKDTree(food)

    def facts(lon, lat):
        near = ftree.query_ball_point((lon * 0.87, lat), 1.0 / 111)
        d, i = road_tree.query((lon * 0.87, lat), k=8)
        d = np.atleast_1d(d) * 111
        within = [road_cls[j] for dd, j in zip(d, np.atleast_1d(i)) if dd <= 0.3]
        return {"food_1km": len(near), "road_km": round(float(d[0]), 2),
                "road_cls": int(min(within)) if within else int(road_cls[np.atleast_1d(i)[0]])}

    out = []
    for e, t, lon, lat in lodging:
        name = t.get("name:en") or t.get("name")
        rec = {"name": name, "type": STAY_TYPES[t["tourism"]], "lon": lon, "lat": lat,
               "district": loc.district(lon, lat), "locality": places.locality(lon, lat)[0], "corp": False,
               "osm": f"{e['type']}/{e['id']}", "stars": _num(t.get("stars")), "rooms": _num(t.get("rooms")),
               "meals": "restaurant on site" if t.get("restaurant") == "yes" or t.get("amenity") in FOOD_AMENITIES
               else None,
               "internet": t.get("internet_access") in ("wlan", "yes", "wifi"),
               "wheelchair": t.get("wheelchair") == "yes"}
        if re.search(rf"\b{C.abbr}\b|tourist rest house|\btrh\b", name, re.I):
            rec["corp_osm"] = True
        if C.osm_match and C.osm_match.search(f"{name} {t.get('operator', '')}"):
            rec.update({"corp": True, "type": f"{C.abbr} rest house", "placed": "OpenStreetMap"})
        rec.update(facts(lon, lat))
        out.append(rec)
    # One rest house is sometimes mapped twice (a hotel and a hostel node at
    # the same spot): keep the first within 150 m.
    kept = []
    for r in out:
        if r["corp"] and any(k["corp"] and _hav_km(r["lon"], r["lat"], k["lon"], k["lat"]) < 0.15 for k in kept):
            continue
        kept.append(r)
    out = kept

    placed = []
    for h in houses:
        names = C.place.get(h["slug"], [])
        p = places.find(names, loc, C.district.get(h["slug"])) if names else None
        if p:
            lon, lat, how = p["lon"], p["lat"], f"at {p['name']} (village/town centre)"
        elif h["slug"] in C.fallback:
            lat, lon = C.fallback[h["slug"]]
            how = "approximate"
        else:
            continue
        # Prefer an OSM lodging tagged as the rest house itself: named for the
        # same place within 6 km, or any rest house within 1.5 km.
        keys = [_norm(n) for n in names]
        cands = [r for r in out if r.get("corp_osm") and (
            (_hav_km(lon, lat, r["lon"], r["lat"]) < 6 and any(k and k in _norm(r["name"]) for k in keys))
            or _hav_km(lon, lat, r["lon"], r["lat"]) < 1.5)]
        if cands:
            best = min(cands, key=lambda r: _hav_km(lon, lat, r["lon"], r["lat"]))
            lon, lat, how = best["lon"], best["lat"], "OpenStreetMap"
            out.remove(best)
        district = loc.district(lon, lat) or min(loc.shapes, key=lambda k: loc.shapes[k].distance(
            __import__("shapely.geometry", fromlist=["Point"]).Point(lon, lat)))
        rec = {"name": h["name"], "type": f"{C.abbr} tourist rest house", "lon": lon, "lat": lat,
               "district": district, "locality": places.locality(lon, lat)[0], "corp": True, "slug": h["slug"],
               "placed": how, "url": C.url(h["slug"]), "plan": _meal_plan(h["rooms"]),
               "room_types": len([r for r in h["rooms"] if not re.search(r"plan|\bEP\b|\bCP\b|\bMAP\b|C P", r, re.I)]),
               "meals": None, "stars": None, "rooms": None, "internet": False, "wheelchair": False}
        if rec["plan"] in ("CP", "MAP"):
            rec["meals"] = MEAL_PLANS[rec["plan"]]
        rec.update(facts(lon, lat))
        placed.append(rec)
    for r in out:
        r.pop("corp_osm", None)
    allst = placed + out
    for r in allst:
        r["checks"] = {"road": r["road_km"] <= 0.3 and r["road_cls"] <= 3,
                       "food": bool(r["meals"]) or r["food_1km"] > 0, "state": r["corp"]}
    return allst


# --------------------------------------------------------------------------
# Visitors (UTDB), from the local Uttarakhand point-in-time database
# --------------------------------------------------------------------------

# Built by python -m nowcast.utdb from UTDB's published reports.
PIT = os.environ.get("UTTARAKHAND_PIT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "pit.sqlite"))


def visitors(R):
    """Uttarakhand Tourism Development Board's published yearly visitor
    counts for the region's destinations (public PDFs, as filed in pit.sqlite).
    Without pit.sqlite (the monthly workflow), from data/utdb/<region>.json,
    which a local build with pit.sqlite rewrites: commit it after the board's
    yearly release."""
    import sqlite3
    p = os.path.join(HERE, "data", "utdb", f"{R.id}.json")
    if not os.path.exists(PIT):
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    con = sqlite3.connect(PIT)
    rows = con.execute("select entity, substr(valid_time,1,4), value, ref, meta from obs "
                       "where source='utdb' and series='tourists_total' order by entity, valid_time").fetchall()
    out = {}
    for ent, yr, val, ref, meta in rows:
        if ent not in R.utdb:
            continue
        m = json.loads(meta or "{}")
        name, dist = R.utdb[ent]
        r = out.setdefault(ent, {"name": name, "district": dist, "years": {}, "refs": []})
        r["years"][yr] = {"total": int(val), "foreign": m.get("foreign")}
        if ref and ref not in r["refs"]:
            r["refs"].append(ref)
    res = sorted(out.values(), key=lambda r: -max(v["total"] for v in r["years"].values()))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


# --------------------------------------------------------------------------
# Build the page
# --------------------------------------------------------------------------

# Suggested circuits: overnight bases in order. Each is routed on the page
# itself, the same way as a circuit you build.
KUMAON_CIRCUITS = [
    ("Lakes and the Corbett edge", "Short, gentle drives: the lake towns above Haldwani, then down to the sal forest at Ramnagar.",
     ["Haldwani", "Bhimtal", "Nainital", "Ramnagar", "Haldwani"]),
    ("Almora, Jageshwar and Binsar", "The old Chand capital, the Jageshwar temple group in its deodar valley, the Binsar ridge and Kausani's long view of the snows.",
     ["Haldwani", "Mukteshwar", "Almora", "Jageshwar", "Binsar", "Kausani", "Ranikhet", "Haldwani"]),
    ("Caves and Shaktipeeths of the east", "Champawat's temples, the Lohaghat hills, the Patal Bhuvaneshwar cave and the Chaukori tea-estate ridge.",
     ["Tanakpur", "Champawat", "Lohaghat", "Pithoragarh", "Patal Bhuvaneshwar", "Chaukori", "Bageshwar", "Almora", "Haldwani"]),
    ("To the high peaks: Munsiyari", "A long hill drive to Munsiyari under the Panchachuli, back by the Kali valley at Dharchula.",
     ["Haldwani", "Almora", "Chaukori", "Munsiyari", "Dharchula", "Pithoragarh", "Champawat", "Tanakpur"]),
]


def _grid(frame, lon, lat):
    x, y = frame.to_grid(lon, lat)
    return round(float(x), 1), round(float(y), 1)


def _svg_path(geom, frame, eps):
    from shapely.geometry import MultiPolygon, Polygon
    from .imd_nowcast import _path
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if isinstance(geom, MultiPolygon) else []
    return " ".join(p for p in (_path(list(pg.exterior.coords), frame, eps) for pg in polys) if p)


def _mask(shapes, frame):
    import cv2
    m = np.zeros((frame.gh, frame.gw), np.uint8)
    for g in shapes:
        for pg in (g.geoms if hasattr(g, "geoms") else [g]):
            xs, ys = frame.to_grid(*np.array(pg.exterior.coords).T)
            cv2.fillPoly(m, [np.round(np.c_[xs, ys] * 8).astype(np.int32)], 1, shift=3)
    return m.astype(bool)


def _wc_masks(shapes, box):
    import cv2
    w, s, e, n = box
    H, W = int(round((n - s) / WC_RES)), int(round((e - w) / WC_RES))
    out = {}
    for name, g in shapes.items():
        m = np.zeros((H, W), np.uint8)
        for pg in (g.geoms if hasattr(g, "geoms") else [g]):
            c = np.array(pg.exterior.coords)
            pts = np.c_[(c[:, 0] - w) / WC_RES, (n - c[:, 1]) / WC_RES]
            cv2.fillPoly(m, [np.round(pts * 8).astype(np.int32)], 1, shift=3)
        out[name] = m.astype(bool)
    return out


WC_GROUPS = [("tree", [10]), ("shrub_grass", [20, 30, 100]), ("crop", [40]), ("built", [50]),
             ("bare", [60]), ("snow", [70]), ("water", [80, 90])]


def _forest(cls, frame, inside, box):
    """WorldCover tree-cover share on the GRID, as a green tint."""
    import cv2
    w, s, e, n = box
    tree = cv2.blur((cls == 10).astype(np.float32), (3, 3))
    lon, lat = frame.grid_lonlat()
    fx = ((lon - w) / WC_RES - 0.5).astype(np.float32)
    fy = ((n - lat) / WC_RES - 0.5).astype(np.float32)
    f = cv2.remap(tree, fx, fy, interpolation=cv2.INTER_LINEAR, borderValue=0)
    a = np.zeros(f.shape + (4,), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = 28, 112, 58
    a[..., 3] = np.where(inside, np.clip(f * 120, 0, 120), 0).astype(np.uint8)
    return a


def _hotspot_ok(name):
    # Leave out hotspots named for a private home (eBird allows "X's garden").
    return not re.search(r"['’]s\b|\bresidence\b|\bbackyard\b|\bmy\b|\bhome\b|\bhouse\b(?!.*guest)", name, re.I)


def _encode_roads(edges, node_ids, frame, eps=0.0005):
    from .imd_nowcast import _rdp
    idx = {n: i for i, n in enumerate(node_ids)}
    flat_e, geom, off = [], [], [0]
    for e in edges:
        pts = _rdp(e["geom"], eps)
        xs, ys = frame.to_grid(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        q = np.round(np.c_[xs, ys] * 2).astype(int)
        d = np.diff(q, axis=0, prepend=[[0, 0]])
        geom.extend(d.ravel().tolist())
        off.append(len(geom) // 2)
        flat_e.extend([idx[e["a"]], idx[e["b"]], round(e["min"] * 10), round(e["km"] * 10), e["cls"]])
    return flat_e, geom, off


def _asof(*patterns):
    """When the oldest of these cached files was fetched (ISO date): what
    the page reflects at worst. None if none are cached."""
    import glob
    times = [os.path.getmtime(f) for p in patterns for f in glob.glob(os.path.join(CACHE, p))]
    return time.strftime("%Y-%m-%d", time.localtime(min(times))) if times else None


def _sources_asof(R, KF, vis):
    osm = _asof(R.ckey("osm_poi.json"), R.ckey("osm_places.json"))
    return {
        "osm": osm,
        "roads": _asof(R.ckey("osm_roads.json"), R.ckey("osm_protected.json"), R.ckey("osm_districts.json")),
        "wikidata": _asof(R.ckey("wikidata.json")),
        "ebird": _asof(*(f"culture/ebird/{k}_{code}.json" for code in R.ebird for k in ("spplist", "hotspots"))),
        "gbif": _asof(f"culture/gbif/aves_{KF.z - 2}_*.png"),
        "corp": osm if R.corp.osm_match else _asof(R.corp.cache_glob),
        "gi": _asof(*(f"culture/gi/{a}.html" for a in GI_APPS)),
        "utdb": max((y for v in vis for y in v["years"]), default=None),
    }


def _changes(prev, data, limit=4):
    """What's new or gone since the previous build, for the page's strip.
    Sites and stays are matched by OSM id (or Wikidata id, or rest-house
    page), so a renamed place isn't counted as one gone and one new."""
    if not prev:
        return None

    def diff(old, new, key, score=None):
        o = {key(r): r for r in old}
        n = {key(r): r for r in new}
        add = sorted((n[k] for k in n.keys() - o.keys()), key=score or (lambda r: 0), reverse=True)
        gone = sorted((o[k] for k in o.keys() - n.keys()), key=score or (lambda r: 0), reverse=True)
        return add, gone

    out = {"since": prev.get("generated")}
    add, gone = diff(prev["sites"], data["sites"], lambda s: s.get("o") or s.get("q") or s["n"], lambda s: s.get("s") or 0)
    out["sites"] = {"added": len(add), "removed": len(gone), "names": [s["n"] for s in add if s["k"] != "peak"][:limit]}
    add, gone = diff(prev["stays"], data["stays"], lambda s: s.get("osm") or s.get("url") or s["name"])
    out["stays"] = {"added": len(add), "removed": len(gone),
                    "names": [f"{s['name']} at {s['locality']}" for s in add if s.get("corp")][:limit]}
    pb, nb = prev.get("birds", {}), data["birds"]
    add, _ = diff(pb.get("hotspots", []), nb["hotspots"], lambda h: h["id"], lambda h: h["species"])
    out["birds"] = {"hotspots": len(add), "names": [h["name"] for h in add][:limit],
                    "species": nb["species_total"] - pb.get("species_total", nb["species_total"])}
    old_gi = {g["app"]: g["status"] for g in prev.get("gi", [])}
    out["gi"] = [g["name"] for g in data["gi"] if g["status"] == "Registered" and old_gi.get(g["app"]) != "Registered"]
    old_pa = {p["name"] for p in prev.get("protected", [])}
    out["protected"] = [p["name"] for p in data["protected"] if p["name"] not in old_pa]
    old_years = {y for v in prev.get("visitors", []) for y in v["years"]}
    out["visitor_years"] = sorted({y for v in data["visitors"] for y in v["years"]} - old_years)
    return out


def build(R, out_path, template_path, log=print, prev=None):
    """prev: the previous build's data, to list what changed since."""
    import cv2
    import networkx as nx
    from scipy.spatial import cKDTree
    from . import kumaon
    from .kumaon import _jpg, _png

    KF = R.frame
    log(f"culture ({R.name}): terrain")
    dem_g, shade = kumaon._terrain(KF)

    def elev_at(lon, lat):
        x, y = KF.to_grid(lon, lat)
        return float(dem_g[int(np.clip(y, 0, KF.gh - 1)), int(np.clip(x, 0, KF.gw - 1))])

    from .geo import world_lonlat
    from . import land
    lat_rows = world_lonlat(np.full(KF.gh, KF.x0), KF.y0 + 2 * np.arange(KF.gh) + 1, KF.z)[1]
    k = int(round(2 * FLAT_KM / KF.km_per_grid_px)) | 1
    slope = cv2.blur(land.slope_deg(dem_g, lat_rows, scale=2, z=KF.z).astype(np.float32), (k, k))

    def flat_at(lon, lat):
        x, y = KF.to_grid(lon, lat)
        return bool(slope[int(np.clip(y, 0, KF.gh - 1)), int(np.clip(x, 0, KF.gw - 1))] < FLAT_SLOPE)

    log("culture: districts, roads")
    loc = Locator(R)
    inside = _mask(loc.shapes.values(), KF)
    edges = road_graph(R, loc, elev_at, flat_at)
    node_ids = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    node_ll = {}
    for e in edges:
        node_ll[e["a"]] = e["geom"][0]
        node_ll[e["b"]] = e["geom"][-1]
    nodes_xy = [_grid(KF, *node_ll[n]) for n in node_ids]
    ntree = cKDTree([(node_ll[n][0] * 0.87, node_ll[n][1]) for n in node_ids])
    vpts, vcls = [], []
    for e in edges:
        for p in e["geom"]:
            vpts.append((p[0] * 0.87, p[1]))
            vcls.append(e["cls"])
    vtree, vcls = cKDTree(vpts), np.array(vcls)
    G = nx.Graph()
    for e in edges:
        if not G.has_edge(e["a"], e["b"]) or G[e["a"]][e["b"]]["min"] > e["min"]:
            G.add_edge(e["a"], e["b"], min=e["min"], km=e["km"])

    def snap(lon, lat):
        """Nearest junction (for routing) and distance to the nearest road."""
        _, i = ntree.query((lon * 0.87, lat))
        d, _ = vtree.query((lon * 0.87, lat))
        return i, round(float(d) * 111, 2)

    log("culture: sites, places, stays")
    places = Places(R, loc)
    site_list = sites(R, loc)
    houses = R.corp.fetch()
    stay_list = stays(R, loc, places, vtree, vcls, houses)

    log("culture: eBird, GBIF, WorldCover")
    eb = ebird(R)
    birds = gbif_birds(KF).copy()
    birds[~inside] = 0
    cls = worldcover(R)
    wcm = _wc_masks(loc.shapes, R.box)
    landcover = []
    for name in R.districts:
        v = cls[wcm[name]]
        n = max(1, (v > 0).sum())
        landcover.append({"district": name, **{k: round(float(np.isin(v, codes).sum()) / n * 100, 1)
                                               for k, codes in WC_GROUPS}})
    forest = _forest(cls, KF, inside, R.box)

    log("culture: protected areas, GI, visitors")
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union
    protected = []
    for r in osm_protected(R):
        t = r.get("tags", {})
        name = t.get("name:en") or t.get("name")
        if r["type"] == "way" and r.get("geometry"):
            g = Polygon([(p["lon"], p["lat"]) for p in r["geometry"]]).buffer(0)
        elif r["type"] == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in r.get("members", [])
                     if m.get("type") == "way" and m.get("role") in ("outer", "") and m.get("geometry")]
            ps = list(polygonize(unary_union(lines))) if lines else []
            if not ps:
                continue
            g = unary_union(ps).buffer(0)
        else:
            continue
        inter = g.intersection(loc.union).area * 111 * 96
        if not name or inter < 5 or name in R.pa_skip:
            continue
        protected.append({"name": name, "d": _svg_path(g, KF, 0.002), "km2": round(inter),
                          "kind": "National park" if t.get("boundary") == "national_park" or t.get("protect_class") == "2"
                          else "Wildlife sanctuary or reserve"})
    gi = [dict(g, local=bool(R.gi_words.search(g["name"] + " " + g["area"]))) for g in gi_products()]
    vis = visitors(R)
    if os.path.exists(PIT):
        missing = sorted(set(R.utdb) - {e for e, (n, _) in R.utdb.items() if n in {v["name"] for v in vis}})
        if missing:
            log(f"  UTDB entities not in pit.sqlite: {missing}")

    log("culture: destinations and circuits")
    dest = {}

    def add_dest(name, lon, lat, kind):
        if name in dest:
            return
        if any(_hav_km(lon, lat, d["lon"], d["lat"]) < 1.0 for d in dest.values()):
            return
        i, km = snap(lon, lat)
        x, y = _grid(KF, lon, lat)
        dest[name] = {"name": name, "lon": lon, "lat": lat, "x": x, "y": y, "node": int(i), "snap_km": km,
                      "kind": kind, "district": loc.district(lon, lat)}
        if km > 2:  # a trek: name the place where the road ends
            dest[name]["via"] = places.locality(*node_ll[node_ids[i]])[0]

    for name, lat, lon, _ in R.towns():
        add_dest(name, lon, lat, "town")
    for s in stay_list:
        if s["corp"]:
            nm = R.corp.place[s["slug"]][0] if "slug" in s else s["locality"]
            add_dest(nm, s["lon"], s["lat"], "stay")
    for s in sorted(site_list, key=lambda s: -s["score"]):
        if s["kind"] != "peak" and s["score"] >= 6:
            add_dest(s["name"], s["lon"], s["lat"], s["kind"])
    destinations = sorted(dest.values(), key=lambda d: d["name"])
    names = {d["name"] for d in destinations}
    circuits = []
    for title, blurb, stops in R.circuits:
        missing = [s for s in stops if s not in names]
        if missing:
            log(f"  culture circuit {title}: unknown stops {missing}")
            continue
        legs = []
        for a, b in zip(stops, stops[1:]):
            na, nb = node_ids[dest[a]["node"]], node_ids[dest[b]["node"]]
            t = nx.dijkstra_path_length(G, na, nb, weight="min")
            legs.append(round(t))
        circuits.append({"name": title, "blurb": blurb, "stops": stops, "check_min": legs})
        log(f"  {title}: {' -> '.join(f'{s}' for s in stops)} legs (min) {legs}")

    flat_e, geom, off = _encode_roads(edges, node_ids, KF)

    def site_rec(s):
        x, y = _grid(KF, s["lon"], s["lat"])
        return {"n": s["name"], "k": s["kind"], "x": x, "y": y, "d": s["district"], "s": s["score"],
                "w": s["wiki"], "i": s["image"], "q": s["qid"], "o": s["osm"], "e": s["ele"], "r": s["religion"]}

    def stay_rec(s):
        x, y = _grid(KF, s["lon"], s["lat"])
        r = {k: s.get(k) for k in ("name", "type", "district", "locality", "corp", "osm", "url", "plan", "meals",
                                   "stars", "rooms", "internet", "wheelchair", "food_1km", "road_km", "road_cls",
                                   "checks", "placed", "room_types")}
        r.update({"x": x, "y": y, "lat": round(s["lat"], 4), "lon": round(s["lon"], 4)})
        return {k: v for k, v in r.items() if v is not None and v is not False}

    hotspots = []
    for h in eb["hotspots"]:
        if not _hotspot_ok(h["name"]) or not loc.inside(h["lon"], h["lat"], pad=True):
            continue
        x, y = _grid(KF, h["lon"], h["lat"])
        hotspots.append({"id": h["id"], "name": h["name"], "x": x, "y": y, "species": h["species"],
                         "district": h["district"], "last": (h["last"] or "")[:10]})

    outside = np.zeros((KF.gh, KF.gw, 4), np.uint8)
    outside[~inside] = [238, 240, 234, 120]
    data = {
        "generated": int(time.time()),
        "region": _region_info(R),
        "frame": {"w": KF.w, "h": KF.h, "gw": KF.gw, "gh": KF.gh, "x0": KF.x0, "y0": KF.y0, "z": KF.z,
                  "box": list(R.box), "km_px": KF.km_per_grid_px},
        "images": {"shade": _jpg(shade, 78), "dem": kumaon._dem_png(dem_g), "outside": _png(outside, quantize=4),
                   "forest": _png(forest, quantize=64), "birds": _png(birds, quantize=32)},
        "districts": [{"name": k, "d": _svg_path(v, KF, 0.002)} for k, v in loc.shapes.items()],
        "protected": protected,
        "roads": {"nodes": [v for xy in nodes_xy for v in (round(xy[0] * 10), round(xy[1] * 10))],
                  "edges": flat_e, "geom": geom, "off": off, "names": ROAD_NAMES,
                  "speed_hill": SPEED_HILL, "speed_plains": SPEED_PLAINS, "plains_m": PLAINS_M,
                  "flat_slope": FLAT_SLOPE, "flat_km": FLAT_KM},
        "sites": [site_rec(s) for s in site_list],
        "kinds": KIND_LABEL,
        "stays": [stay_rec(s) for s in stay_list],
        "birds": {"districts": eb["districts"], "species_total": eb["species_total"], "hotspots": hotspots},
        "landcover": landcover,
        "gi": [g for g in gi if g["status"]],
        "visitors": vis,
        "destinations": [{k: d[k] for k in ("name", "x", "y", "node", "snap_km", "kind", "district", "via") if k in d}
                         for d in destinations],
        "circuits": circuits,
        "asof": _sources_asof(R, KF, vis),
    }
    data["changes"] = _changes(prev and _upgrade(prev, R), data)
    size = render(data, out_path, template_path)
    last = os.path.join(CACHE, R.ckey("culture_last.json"))
    os.makedirs(os.path.dirname(last), exist_ok=True)
    with open(last, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    return (f"{R.name}: {size / 1e6:.1f} MB, {len(data['sites'])} sites, {len(data['stays'])} stays "
            f"({sum(s['corp'] for s in stay_list)} {R.corp.abbr}), {len(edges)} road links, {len(hotspots)} eBird hotspots, "
            f"{len(protected)} protected areas, {len(destinations)} destinations, {len(circuits)} circuits")


def _region_info(R):
    """What the page needs to know about its region."""
    return {"id": R.id, "name": R.name, "districts": list(R.districts),
            "corp": {"abbr": R.corp.abbr, "name": R.corp.name, "site": R.corp.site, "osm": bool(R.corp.osm_match)},
            "shade_alt": R.shade_alt, "gi_note": R.gi_note, "treks": R.treks, "store": R.store, "nowcast": R.nowcast,
            "back": [list(b) for b in R.back]}


def _upgrade(data, R):
    """Bring a culture_last.json from before regions up to date, so
    --render works without a rebuild."""
    if "region" in data:
        return data
    data = {"generated": data["generated"], "region": _region_info(R), **data}
    for s in data["stays"]:
        if "kmvn" in s:
            s["corp"] = s.pop("kmvn")
    data["gi"] = [{("local" if k == "kumaon" else k): v for k, v in g.items()} for g in data["gi"]]
    return data


def _and(names):
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)


def render(data, out_path, template_path):
    from html import escape
    G = data["region"]
    tpl = open(template_path, encoding="utf-8").read()
    for k, v in {"REGION": escape(G["name"]), "CORP": escape(G["corp"]["abbr"]), "DISTRICTS": escape(_and(G["districts"])),
                 "SHADE_ALT": escape(G["shade_alt"]),
                 "BACK": "".join(f'<a href="{escape(h)}">← {escape(t)}</a>' for h, t in G["back"])}.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    body = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            f'<meta name="description" content="{escape(G["name"])} cultural atlas: temples, forts, peaks, forests, birds, GI '
            'products, visitor numbers, and places to stay along drive-time circuits you plan yourself.">'
            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + body + "</body></html>")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return len(html)


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------

def _kumaon_towns():
    from .kumaon import TOWNS
    return TOWNS


KMVN = Corp(
    abbr="KMVN", name="Kumaon Mandal Vikas Nigam", site="kmvn.in", fetch=kmvn_houses,
    url=lambda slug: f"https://kmvn.in/hotels/{slug}",
    place=KMVN_PLACE, fallback=KMVN_FALLBACK, district=KMVN_DISTRICT, cache_glob="culture/kmvn/*.html")

KUMAON = Region(
    id="kumaon", name="Kumaon", box=(78.6, 28.6, 81.2, 30.7),  # the Kumaon nowcast frame
    districts=["Nainital", "Almora", "Bageshwar", "Champawat", "Pithoragarh", "Udham Singh Nagar"],
    ebird={"IN-UL-NA": "Nainital", "IN-UL-AL": "Almora", "IN-UL-BA": "Bageshwar",
           "IN-UL-CP": "Champawat", "IN-UL-PI": "Pithoragarh", "IN-UL-US": "Udham Singh Nagar"},
    towns=_kumaon_towns, corp=KMVN,
    gi_words=re.compile(r"kumaon|almora|nainital|munsyari|munsiari|berinag|ramnagar|ramgarh|champawat|kausani|"
                        r"pithoragarh|bageshwar|lohaghat|dharchula|udham singh nagar|haldwani", re.I),
    utdb={"nainital": ("Nainital", "Nainital"), "almora": ("Almora", "Almora"),
          "ranikheth": ("Ranikhet", "Almora"), "kausani_bageshwar": ("Kausani and Bageshwar", "Bageshwar"),
          "pithoragarh": ("Pithoragarh", "Pithoragarh"), "champawat": ("Champawat", "Champawat"),
          "kathgodam": ("Kathgodam", "Nainital"), "corbett_national_park": ("Corbett National Park", "Nainital"),
          "udham_singh_nagar": ("Udham Singh Nagar", "Udham Singh Nagar")},
    circuits=KUMAON_CIRCUITS,
    shade_alt="Shaded relief of Kumaon, from the terai plains to the Panchachuli and Nanda Kot",
    gi_note="many (Aipan, Tamta copperware, Bhotia dann and thulma, Ringal) are crafts long practised in Kumaon",
    treks="Pindari, Milam, Kafni, Traill's",
    # Pilibhit Tiger Reserve is in Uttar Pradesh; its OSM outline overlaps
    # Udham Singh Nagar's along the state border.
    pa_skip=frozenset({"Pilibhit Tiger Reserve"}),
    # Wikidata items whose coordinates put them in the wrong place (Patal
    # Bhuvaneshwar 50 km off; Khaptad Lake is in Nepal); OSM has the first.
    wd_bad_coords=frozenset({"Patal Bhuvaneshwar", "Khaptad Lake"}),
    store="kca:", back=(("../", "Kumaon nowcast"), ("../../", "Himalaya Nowcast Atlas")), nowcast="../")

# Always-routable towns, at their OSM place nodes.
GARHWAL_TOWNS = [
    ("Dehradun", 30.326, 78.044, "Dehradun"), ("Mussoorie", 30.457, 78.078, "Dehradun"),
    ("Vikasnagar", 30.469, 77.771, "Dehradun"), ("Chakrata", 30.701, 77.870, "Dehradun"),
    ("Rishikesh", 30.109, 78.292, "Dehradun"), ("Haridwar", 29.938, 78.145, "Haridwar"),
    ("Roorkee", 29.869, 77.890, "Haridwar"), ("Kotdwar", 29.746, 78.520, "Pauri Garhwal"),
    ("Lansdowne", 29.838, 78.682, "Pauri Garhwal"), ("Pauri", 30.148, 78.776, "Pauri Garhwal"),
    ("Srinagar", 30.223, 78.790, "Pauri Garhwal"), ("Devprayag", 30.146, 78.599, "Tehri Garhwal"),
    ("New Tehri", 30.374, 78.433, "Tehri Garhwal"), ("Chamba", 30.346, 78.394, "Tehri Garhwal"),
    ("Dhanaulti", 30.425, 78.246, "Tehri Garhwal"), ("Uttarkashi", 30.729, 78.440, "Uttarkashi"),
    ("Barkot", 30.809, 78.206, "Uttarkashi"), ("Purola", 30.879, 78.083, "Uttarkashi"),
    ("Harsil", 31.037, 78.751, "Uttarkashi"), ("Gangotri", 30.994, 78.940, "Uttarkashi"),
    ("Rudraprayag", 30.287, 78.981, "Rudraprayag"), ("Ukhimath", 30.514, 79.095, "Rudraprayag"),
    ("Gaurikund", 30.653, 79.027, "Rudraprayag"), ("Gopeshwar", 30.409, 79.319, "Chamoli"),
    ("Karnaprayag", 30.260, 79.217, "Chamoli"), ("Gauchar", 30.289, 79.157, "Chamoli"),
    ("Joshimath", 30.556, 79.561, "Chamoli"), ("Badrinath", 30.742, 79.493, "Chamoli"),
]

GARHWAL_CIRCUITS = [
    ("Doon, Mussoorie and Tehri", "Up from the Doon valley to Mussoorie's ridge, along the Dhanaulti deodars to Chamba and the Tehri lake, and down to Rishikesh.",
     ["Dehradun", "Mussoorie", "Dhanaulti", "Chamba", "New Tehri", "Rishikesh", "Dehradun"]),
    ("The prayags to Badrinath", "Up the Alaknanda by its confluences: Devprayag, Rudraprayag, Karnaprayag, Nandprayag and Vishnuprayag below Joshimath, to Badrinath.",
     ["Rishikesh", "Devprayag", "Srinagar", "Rudraprayag", "Karnaprayag", "Joshimath", "Badrinath"]),
    ("Kedar valley and Tungnath", "To Gaurikund, where the Kedarnath trail starts, then over the Chopta meadows below Tungnath to Gopeshwar.",
     ["Rishikesh", "Rudraprayag", "Gaurikund", "Ukhimath", "Tunganath", "Gopeshwar", "Karnaprayag", "Srinagar", "Rishikesh"]),
    ("Yamunotri and Gangotri", "Over Mussoorie to the Yamuna valley and the Yamunotri trail, then across to Uttarkashi and up the Bhagirathi past Harsil to Gangotri.",
     ["Dehradun", "Mussoorie", "Barkot", "Yamunotri", "Barkot", "Uttarkashi", "Harsil", "Gangotri"]),
    ("The Pauri ridge", "From Kotdwar up to the cantonment at Lansdowne, along the oak ridges to Pauri, down to Srinagar and by Devprayag to Rishikesh.",
     ["Kotdwar", "Lansdowne", "Pauri", "Srinagar", "Devprayag", "Rishikesh"]),
    ("Jaunsar-Bawar and the Tons", "Up from Vikasnagar to Chakrata's cantonment, through the Jaunsari villages to Purola, and back by the Yamuna valley and Mussoorie.",
     ["Dehradun", "Vikasnagar", "Chakrata", "Purola", "Barkot", "Mussoorie", "Dehradun"]),
]

# gmvnonline.com sits behind a bot challenge that turns away scripted
# requests, robots.txt included, so GMVN's rest houses are the ones mapped
# on OpenStreetMap (named for GMVN, or with it as operator): a partial list.
GMVN = Corp(
    abbr="GMVN", name="Garhwal Mandal Vikas Nigam", site="gmvnonline.com", fetch=list,
    url=lambda slug: "https://gmvnonline.com/", place={},
    osm_match=re.compile(r"\bgmvn\b|garh?wal mandal", re.I))

GARHWAL = Region(
    id="garhwal", name="Garhwal",
    box=(77.5, 29.35, 80.2, 31.55),  # OSM's district outlines span 77.57-80.10 E, 29.43-31.46 N
    districts=["Uttarkashi", "Chamoli", "Rudraprayag", "Tehri Garhwal", "Pauri Garhwal", "Dehradun", "Haridwar"],
    ebird={"IN-UL-UT": "Uttarkashi", "IN-UL-CL": "Chamoli", "IN-UL-RP": "Rudraprayag", "IN-UL-TG": "Tehri Garhwal",
           "IN-UL-PG": "Pauri Garhwal", "IN-UL-DD": "Dehradun", "IN-UL-HA": "Haridwar"},
    towns=lambda: GARHWAL_TOWNS, corp=GMVN,
    gi_words=re.compile(r"garhwal|uttarkashi|chamoli|rudraprayag|tehri|pauri|dehradun|haridwar|mussoorie|"
                        r"rishikesh|jaunsar|chakrata|purola|harsil|joshimath|badrinath|kedarnath|gangotri", re.I),
    # UTDB's 19 Garhwal destinations (python -m nowcast.utdb). Some are wider
    # than the name: Tehri is the district from 2025, Kotdwar includes
    # Swargashram and Chilla, Rudraprayag excludes Kedarnath.
    utdb={"dehradun": ("Dehradun", "Dehradun"), "rishikesh": ("Rishikesh", "Dehradun"),
          "mussoorie": ("Mussoorie", "Dehradun"), "haridwar": ("Haridwar", "Haridwar"),
          "pauri": ("Pauri", "Pauri Garhwal"), "srinagar": ("Srinagar", "Pauri Garhwal"),
          "kotdwar": ("Kotdwar, Swargashram and Chilla", "Pauri Garhwal"),
          "rudraprayag": ("Rudraprayag (without Kedarnath)", "Rudraprayag"),
          "kedarnath": ("Kedarnath", "Rudraprayag"),
          "gopeshwar": ("Gopeshwar, Nandprayag and Tharali", "Chamoli"),
          "joshimath": ("Joshimath, Govindghat and Ghangaria", "Chamoli"),
          "badrinath": ("Badrinath", "Chamoli"), "auli": ("Auli", "Chamoli"),
          "hemkund_sahib": ("Hemkund Sahib", "Chamoli"), "valley_of_flowers": ("Valley of Flowers", "Chamoli"),
          "tehri": ("Tehri", "Tehri Garhwal"),
          "uttarkashi": ("Uttarkashi, Harsil and Gangnani", "Uttarkashi"),
          "gangotri": ("Gangotri", "Uttarkashi"), "yamunotri": ("Yamunotri", "Uttarkashi")},
    circuits=GARHWAL_CIRCUITS,
    shade_alt="Shaded relief of Garhwal, from the Doon valley and the Haridwar plains to Bandarpunch, "
              "the Gangotri peaks and Nanda Devi",
    gi_note="many (Ringal craft, Bichhu Buti nettle fabric, Thulma and Bhotia dann; Jhangora, Mandua and "
            "Purola's Lal Chawal) are long part of Garhwal's crafts and hill farming",
    treks="Gomukh, Kedartal, Satopanth, Roopkund",
    # Hastinapur sanctuary is in Uttar Pradesh; its OSM outline overlaps
    # Haridwar's along the state border.
    pa_skip=frozenset({"Hastinapur WLS"}),
    # Amangarh Tiger Reserve is across the line in Uttar Pradesh.
    wd_bad_coords=frozenset({"Amangarh Tiger Reserve"}),
    cache_dir="garhwal", store="gca:", back=(("../../", "Himalaya Nowcast Atlas"),))

REGIONS = {R.id: R for R in (KUMAON, GARHWAL)}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(HERE, "culture_template.html")


def out_path(R):
    return os.path.join(HERE, "static", R.id, "culture", "index.html")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build a regional cultural atlas into static/<region>/culture/.")
    ap.add_argument("--region", default="kumaon", help=f"comma-separated: {', '.join(REGIONS)}")
    ap.add_argument("--render", action="store_true", help="re-render from the last build's data (template work)")
    ap.add_argument("--publish", metavar="DIR", help="also copy each page and its data to DIR/<region>/culture/")
    args = ap.parse_args()
    for rid in args.region.split(","):
        R = REGIONS[rid.strip()]
        if args.render:
            data = json.load(open(os.path.join(CACHE, R.ckey("culture_last.json")), encoding="utf-8"))
            n = render(_upgrade(data, R), out_path(R), TEMPLATE)
            print(f"wrote {out_path(R)} ({n / 1e6:.1f} MB)")
        else:
            # The previous build: what's published (the workflow), else the last local build.
            prev = None
            for p in ([os.path.join(args.publish, R.id, "culture", "data.json")] if args.publish else []) + \
                     [os.path.join(CACHE, R.ckey("culture_last.json"))]:
                if os.path.exists(p):
                    prev = json.load(open(p, encoding="utf-8"))
                    break
            print(build(R, out_path(R), TEMPLATE, prev=prev))
        if args.publish:
            import shutil
            dst = os.path.join(args.publish, R.id, "culture")
            os.makedirs(dst, exist_ok=True)
            shutil.copy(out_path(R), os.path.join(dst, "index.html"))
            shutil.copy(os.path.join(CACHE, R.ckey("culture_last.json")), os.path.join(dst, "data.json"))
