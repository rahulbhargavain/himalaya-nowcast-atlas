"""Point-in-time archive: what each build knew, stored as it was known.

Every build writes dist/snapshot/ (see write_snapshot). The GitHub
workflow then files it into the `data` branch with

    python -m nowcast.archive dist/snapshot <data-branch checkout>

which copies the snapshot to snapshots/YYYY/MM/DD/HHMMZ/ and appends rows
to cumulative CSV tables at the branch root. Nothing already archived is
ever rewritten, so later analysis can only see what was available at each
build time (no lookahead).

Tables (one row per build x item; `build_utc` is when the build ran,
`valid_utc` is the time the value describes):

  towns.csv     radar dBZ now, extrapolated onset, model rain, snow line
  forecast.csv  per town and hour: model rain and snow line as forecast
  stations.csv  airport METAR observations (deduplicated by report time)
  gauges.csv    GloFAS discharge now, % of normal, and the 7-day forecast
  builds.csv    one row per build: echo area, motion, fires, counts

Radar grids (half of the atlas GRID, uint8 dBZ, 0 = no echo): the latest
observed frame and the extrapolation for +1, +2, +4 and +8 h, so the
nowcast can be scored against later observed frames.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

LEADS_H = (1, 2, 4, 8)


def _utc(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _grid_png(path, dbz):
    small = dbz[::2, ::2]
    Image.fromarray(np.clip(np.round(small), 0, 90).astype(np.uint8), "L").save(path, optimize=True)


def write_snapshot(out_dir, data, frames_all, t_all, t0_index, step_min):
    """Write the build's point-in-time record to out_dir (replaced each build)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    built = data["generated"]
    t0 = data["t0"]
    m = data["model"]
    rec = {
        "schema": 1,
        "build_utc": _utc(built), "radar_t0_utc": _utc(t0),
        "model_times_utc": [_utc(t) for t in m["times"]],
        "motion": data["radar"]["motion"], "cells": data["radar"]["cells"],
        "events": data["events"],
        "towns": [{k: t[k] for k in ("name", "dbz_now", "radar_onset_min", "model_rain_8h", "model_rain", "snowline", "dbz_series")}
                  for t in data["towns"]],
        "pressure_centres": [{"valid_utc": _utc(t), "centres": [{k: c[k] for k in ("kind", "hpa", "lat", "lon")} for c in h["centres"]]}
                             for t, h in zip(m["times"], m["pressure"])],
        "snowline_mean_m": [{"valid_utc": _utc(t), "m": s["mean_m"]} for t, s in zip(m["times"], m["snowline"])],
        "stations": [{k: s.get(k) for k in ("id", "name", "elev", "t", "td", "wdir", "wspd", "qnh", "wx", "vis", "cover", "obs", "raw")}
                     for s in data["stations"]],
        "gauges": [{k: g.get(k) for k in ("river", "place", "lat", "lon", "q_now", "pct_of_median", "times", "q", "today_index")}
                   for g in data["water"]["gauges"]],
        "fires": {k: v for k, v in (data.get("hkh", {}).get("fires") or {}).items() if k != "src"},
    }
    with gzip.open(os.path.join(out_dir, "snapshot.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))

    _grid_png(os.path.join(out_dir, "radar_obs.png"), frames_all[t0_index])
    per_h = 60 // step_min
    for h in LEADS_H:
        k = t0_index + h * per_h
        if k < len(frames_all):
            _grid_png(os.path.join(out_dir, f"radar_fc_{h}h.png"), frames_all[k])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"build_utc": rec["build_utc"], "radar_t0_utc": rec["radar_t0_utc"],
                   "grid": "half of atlas GRID (z7/4), uint8 dBZ", "leads_h": list(LEADS_H)}, f)
    return rec


def _append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def _seen_station_reports(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["id"], r["obs_utc"]) for r in csv.DictReader(f)}


def file_snapshot(snap_dir, root):
    """Copy a snapshot into the archive checkout and append the CSV tables."""
    with gzip.open(os.path.join(snap_dir, "snapshot.json.gz"), "rt", encoding="utf-8") as f:
        rec = json.load(f)
    b = dt.datetime.strptime(rec["build_utc"], "%Y-%m-%dT%H:%MZ")
    dest = os.path.join(root, "snapshots", b.strftime("%Y"), b.strftime("%m"), b.strftime("%d"), b.strftime("%H%MZ"))
    if os.path.exists(dest):
        print(f"already archived: {dest}")
        return False
    shutil.copytree(snap_dir, dest)

    B = rec["build_utc"]
    _append(os.path.join(root, "builds.csv"),
            ["build_utc", "radar_t0_utc", "echo_km2", "motion_heading", "motion_kmh", "cells", "stations", "gauges", "fire_pixels"],
            [[B, rec["radar_t0_utc"], (rec["motion"] or {}).get("echo_km2", 0), (rec["motion"] or {}).get("heading"),
              (rec["motion"] or {}).get("speed_kmh"), len(rec["cells"]), len(rec["stations"]), len(rec["gauges"]),
              rec["fires"].get("pixels")]])
    mt = rec["model_times_utc"]
    _append(os.path.join(root, "towns.csv"),
            ["build_utc", "radar_t0_utc", "town", "dbz_now", "radar_onset_min", "model_rain_8h_mm"],
            [[B, rec["radar_t0_utc"], t["name"], t["dbz_now"], t["radar_onset_min"], t["model_rain_8h"]] for t in rec["towns"]])
    _append(os.path.join(root, "forecast.csv"),
            ["build_utc", "valid_utc", "lead_h", "town", "model_rain_mm", "snowline_m"],
            [[B, v, round((dt.datetime.strptime(v, "%Y-%m-%dT%H:%MZ") - b).total_seconds() / 3600, 2), t["name"], r, s]
             for t in rec["towns"] for v, r, s in zip(mt, t["model_rain"], t["snowline"])])
    seen = _seen_station_reports(os.path.join(root, "stations.csv"))
    rows = []
    for s in rec["stations"]:
        if s.get("obs") is None:
            continue
        key = (s["id"], _utc(s["obs"]))
        if key in seen:
            continue
        rows.append([B, key[1], s["id"], s["name"], s["elev"], s["t"], s["td"], s["wdir"], s["wspd"], s["qnh"], s["wx"], s["vis"], s["cover"], s["raw"]])
    _append(os.path.join(root, "stations.csv"),
            ["build_utc", "obs_utc", "id", "name", "elev_m", "temp_c", "dewpoint_c", "wind_dir", "wind_kt", "qnh_hpa", "wx", "vis", "cover", "raw"], rows)
    grows = []
    for g in rec["gauges"]:
        ti = g.get("today_index") or 0
        fc = (g.get("q") or [])[ti:ti + 8]
        grows.append([B, g["river"], g["place"], g["lat"], g["lon"], g["q_now"], g["pct_of_median"],
                      (g.get("times") or [None])[ti] if g.get("times") else None, json.dumps(fc)])
    _append(os.path.join(root, "gauges.csv"),
            ["build_utc", "river", "place", "lat", "lon", "q_now_m3s", "pct_of_median", "date", "q_forecast_from_date"], grows)
    print(f"archived {B} -> {dest} ({len(rows)} new station reports)")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m nowcast.archive <snapshot dir> <archive checkout>")
    file_snapshot(sys.argv[1], sys.argv[2])
