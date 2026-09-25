"""Winter chill in the Himalayan apple belts, 1990-91 to 2025-26.

Apples need a cold winter to break dormancy: the National Horticulture
Board puts Himalayan apple growing at 1,500-2,700 m, "which experience
1,000-1,500 hours of chilling" (hours at or below 7 C). This script
measures those chill hours for every winter since 1990-91 in eight apple
districts, and finds the elevation below which a winter no longer reaches
1,000 chill hours (the "chill line"), to see how far it has moved uphill.

Data: ERA5-Land hourly 2 m temperature (0.1 deg) from the Open-Meteo
historical archive, downscaled by Open-Meteo to a common reference
elevation of 2,000 m. Other elevations are derived with a fixed lapse
rate (6.5 C/km, with 5.0 C/km as a sensitivity check, since winter lapse
rates in the Himalaya are often shallower). So the trend over time is
real reanalysis signal; the elevation profile is a modelled assumption.

Winter = 1 November to the end of February.

    python analysis/chill.py        # writes analysis/chill_*.csv and chill_report.html
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nowcast.geo import get_json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REGIONS = [
    ("Shopian (Kashmir)", 33.72, 74.83),
    ("Sopore (Kashmir)", 34.30, 74.47),
    ("Kullu (Himachal)", 31.96, 77.11),
    ("Kotkhai, Shimla (Himachal)", 31.12, 77.53),
    ("Kinnaur (Himachal)", 31.55, 78.25),
    ("Harsil (Uttarakhand)", 31.04, 78.74),
    ("Mukteshwar (Uttarakhand)", 29.47, 79.65),
    ("Jumla (Nepal)", 29.27, 82.18),
]
REF_ELEV = 2000
FIRST_WINTER, LAST_WINTER = 1990, 2025          # winter labelled by its November year
CHILL_C = 7.2
THRESHOLDS = (1000, 1200)
ELEVS = np.arange(800, 3601, 50)


def fetch_winter(year):
    """Hourly temperature at REF_ELEV for all regions, Nov `year` - Feb `year+1`."""
    end = f"{year + 1}-02-29" if (year + 1) % 4 == 0 else f"{year + 1}-02-28"
    j = get_json("https://archive-api.open-meteo.com/v1/archive", params=dict(
        latitude=",".join(str(r[1]) for r in REGIONS), longitude=",".join(str(r[2]) for r in REGIONS),
        elevation=",".join(str(REF_ELEV) for _ in REGIONS),
        start_date=f"{year}-11-01", end_date=end, hourly="temperature_2m", models="era5_land", timezone="GMT"),
        cache_key=f"chill/era5land_{year}.json", timeout=180)
    j = j if isinstance(j, list) else [j]
    return [np.array([np.nan if v is None else v for v in r["hourly"]["temperature_2m"]], float) for r in j]


def chill_hours(t, lapse_c_per_km, elev):
    te = t - lapse_c_per_km * (elev - REF_ELEV) / 1000.0
    return int(np.sum(te[np.isfinite(te)] <= CHILL_C))


def chill_line(t, lapse, threshold):
    """Lowest elevation (50 m steps) where the winter reaches `threshold` chill hours."""
    for e in ELEVS:
        if chill_hours(t, lapse, e) >= threshold:
            return int(e)
    return None


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    b = np.polyfit(x, y, 1)
    res = y - np.polyval(b, x)
    se = np.sqrt(res @ res / (n - 2) / np.sum((x - x.mean()) ** 2))
    return b[0], se, n


def mann_kendall_p(y):
    """Two-sided Mann-Kendall trend test p-value (no tie correction)."""
    from math import erf, sqrt
    y = np.asarray([v for v in y if v is not None and np.isfinite(v)], float)
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / sqrt(var) if s != 0 else 0.0
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def main():
    winters = list(range(FIRST_WINTER, LAST_WINTER + 1))
    temps = {}
    for y in winters:
        temps[y] = fetch_winter(y)
        time.sleep(0.3)
        print(f"winter {y}-{str(y + 1)[2:]} fetched", flush=True)

    rows, summary = [], []
    for k, (name, lat, lon) in enumerate(REGIONS):
        series = {"winter": [], "chill_2000": [], "mean_t_2000": []}
        lines = {(lapse, th): [] for lapse in (6.5, 5.0) for th in THRESHOLDS}
        for y in winters:
            t = temps[y][k]
            ch = chill_hours(t, 6.5, REF_ELEV)
            series["winter"].append(y)
            series["chill_2000"].append(ch)
            series["mean_t_2000"].append(float(np.nanmean(t)))
            row = {"region": name, "winter": f"{y}-{str(y + 1)[2:]}", "chill_hours_2000m": ch,
                   "mean_winter_temp_2000m": round(float(np.nanmean(t)), 2)}
            for (lapse, th) in lines:
                v = chill_line(t, lapse, th)
                lines[(lapse, th)].append(v)
                row[f"chill_line_{th}h_lapse{lapse}"] = v
            rows.append(row)
        s_ch, se_ch, n = ols(winters, series["chill_2000"])
        s_t, se_t, _ = ols(winters, series["mean_t_2000"])
        out = {"region": name, "lat": lat, "lon": lon, "n_winters": n,
               "chill_2000m_per_decade": round(10 * s_ch), "chill_2000m_per_decade_se": round(10 * se_ch),
               "chill_2000m_mk_p": round(mann_kendall_p(series["chill_2000"]), 4),
               "winter_temp_per_decade": round(10 * s_t, 2), "winter_temp_per_decade_se": round(10 * se_t, 2),
               "chill_2000m_1990s": round(np.mean(series["chill_2000"][:10])),
               "chill_2000m_2016_25": round(np.mean(series["chill_2000"][-10:])),
               "series": series}
        for (lapse, th), v in lines.items():
            vv = [np.nan if x is None else x for x in v]
            sl, se, _ = ols(winters, vv)
            out[f"line_{th}_{lapse}"] = {
                "per_decade_m": round(10 * sl), "se_m": round(10 * se),
                "mean_1990s": round(float(np.nanmean(vv[:10]))), "mean_2016_25": round(float(np.nanmean(vv[-10:]))),
                "mk_p": round(mann_kendall_p(vv), 4), "series": v}
        summary.append(out)
        l = out["line_1000_6.5"]
        print(f"{name:28s} chill@2000m {out['chill_2000m_1990s']:>5} -> {out['chill_2000m_2016_25']:>5} h "
              f"({out['chill_2000m_per_decade']:+} h/decade, p={out['chill_2000m_mk_p']}); "
              f"1000h line {l['mean_1990s']} -> {l['mean_2016_25']} m ({l['per_decade_m']:+} m/decade)")

    import csv
    with open(os.path.join(HERE, "chill_by_winter.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "chill_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ref_elev": REF_ELEV, "chill_c": CHILL_C, "first": FIRST_WINTER, "last": LAST_WINTER,
                   "regions": summary}, f, indent=1)
    print("wrote analysis/chill_by_winter.csv and chill_summary.json")


if __name__ == "__main__":
    main()
