"""Render analysis/chill_summary.json as a small self-contained HTML report."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def spark(series, trend_per_decade, w=300, h=90, lo=None, hi=None, color="#2d6fa3"):
    ys = [v for v in series if v is not None]
    lo = min(ys) if lo is None else lo
    hi = max(ys) if hi is None else hi
    rng = (hi - lo) or 1
    n = len(series)
    X = lambda i: 4 + i / (n - 1) * (w - 8)
    Y = lambda v: h - 14 - (v - lo) / rng * (h - 24)
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(series) if v is not None)
    mean = sum(ys) / len(ys)
    mid = (n - 1) / 2
    t0 = mean - trend_per_decade / 10 * mid
    t1 = mean + trend_per_decade / 10 * mid
    return (f'<svg viewBox="0 0 {w} {h}" class="sp" role="img"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/>'
            f'<line x1="{X(0):.1f}" y1="{Y(t0):.1f}" x2="{X(n - 1):.1f}" y2="{Y(t1):.1f}" stroke="var(--accent)" stroke-width="1.6" stroke-dasharray="4 3"/>'
            f'<text x="4" y="{h - 2}" font-size="10" fill="var(--ink2)">1990–91</text><text x="{w - 4}" y="{h - 2}" font-size="10" fill="var(--ink2)" text-anchor="end">2025–26</text></svg>')


def main():
    d = json.load(open(os.path.join(HERE, "chill_summary.json"), encoding="utf-8"))
    cards, rows = [], []
    for r in d["regions"]:
        L = r["line_1000_6.5"]
        cards.append(f"""<section class="card"><h3>{r['region']}</h3>
<div class="k">Chill hours at 2,000 m: <b>{r['chill_2000m_1990s']:,}</b> (1990s) → <b>{r['chill_2000m_2016_25']:,}</b> (2016–25),
{r['chill_2000m_per_decade']:+} h per decade{' <span class="sig">clear trend</span>' if r['chill_2000m_mk_p'] < 0.05 else ''}</div>
{spark(r['series']['chill_2000'], r['chill_2000m_per_decade'])}
<div class="k">1,000-hour chill line: <b>{L['mean_1990s']:,} m</b> → <b>{L['mean_2016_25']:,} m</b> ({L['per_decade_m']:+} m per decade){' (at or below the lowest elevation tested)' if L['mean_2016_25'] <= 850 else ''}</div></section>""")
        rows.append(f"<tr><td>{r['region']}</td><td class='n'>{r['winter_temp_per_decade']:+.2f} ± {r['winter_temp_per_decade_se']:.2f}</td>"
                    f"<td class='n'>{r['chill_2000m_per_decade']:+}</td><td class='n'>{r['chill_2000m_mk_p']:.2f}</td>"
                    f"<td class='n'>{L['per_decade_m']:+}</td><td class='n'>{r['line_1000_5.0']['per_decade_m']:+}</td><td class='n'>{r['line_1200_6.5']['per_decade_m']:+}</td></tr>")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Himalayan Apple Chill</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Alegreya:wght@700&family=Alegreya+Sans:wght@400;500;700&family=IBM+Plex+Mono&display=swap">
<style>
:root{{--paper:#eef0ea;--panel:#f7f8f4;--ink:#1f2a2e;--ink2:#4d5a5c;--rule:#cfd4cb;--accent:#b8324f}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--paper:#11171a;--panel:#182024;--ink:#e3e8e2;--ink2:#a3aea9;--rule:#2c373b;--accent:#e0607a}}}}
:root[data-theme="dark"]{{color-scheme:dark;--paper:#11171a;--panel:#182024;--ink:#e3e8e2;--ink2:#a3aea9;--rule:#2c373b;--accent:#e0607a}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 "Alegreya Sans",system-ui,sans-serif;padding:20px 16px 48px}}
main{{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:16px}}
h1{{font:700 34px/1.1 Alegreya,Georgia,serif;margin:0;text-wrap:balance}}
h2{{font:700 20px/1.2 Alegreya,Georgia,serif;margin:8px 0 0}}
h3{{margin:0;font-size:17px}}
p{{margin:0;max-width:72ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:12px;display:flex;flex-direction:column;gap:6px}}
.k{{font-size:14px;color:var(--ink2)}} .k b{{color:var(--ink)}}
.sig{{font-size:12px;border:1px solid var(--accent);color:var(--accent);border-radius:999px;padding:0 6px}}
.sp{{width:100%;height:auto;display:block}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:left}}
th{{font-size:13px;color:var(--ink2)}} td.n{{text-align:right;font-family:"IBM Plex Mono",monospace;font-size:13px}}
.note{{font-size:14px;color:var(--ink2)}}
</style></head><body><main>
<h1>Winter chill in the Himalayan apple belts, 1990–2026</h1>
<p>Apples need cold winters: roughly 1,000–1,500 hours at or below 7 °C to flower and fruit well (National Horticulture Board). This measures those hours for every winter (November–February) since 1990–91 in eight apple districts, from ERA5-Land hourly temperature, and tracks the elevation below which a winter no longer reaches 1,000 hours.</p>
<h2>What it shows</h2>
<p>Winters have warmed in every district, from +0.05 to +0.54 °C per decade, and chill hours have fallen everywhere. In six districts the 1,000-hour line has moved uphill by about 20–60 m per decade, 75–165 m between the 1990s and 2016–25; in Sopore and Kinnaur it sits at or below the lowest elevation tested, so it can't show movement. The trend is statistically clear on its own only in Kullu and Jumla (Nepal); elsewhere it points the same way but is within year-to-year noise. The districts rise and fall together (average correlation 0.64), so the eight series are not eight independent confirmations. Pooled, the decline is borderline (p ≈ 0.08).</p>
<p>That is consistent with reports of apple growing moving to higher ground (toward Kinnaur and Lahaul–Spiti, away from lower Shimla and Kullu orchards), but this data alone can't show the shift in orchards or yields.</p>
<div class="grid">{''.join(cards)}</div>
<h2>Trends by district</h2>
<div class="scroll"><table><thead><tr><th>District</th><th>Winter temp °C/decade</th><th>Chill h/decade (2,000 m)</th><th>trend p</th><th>1,000 h line m/decade</th><th>same, 5 °C/km lapse</th><th>1,200 h line m/decade</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Limits</h2>
<p class="note">ERA5-Land (0.1°, about 9 km) runs warm in steep terrain: at Mukteshwar it gives a winter mean near 10 °C at 2,000 m where the station records closer to 7–8 °C, so absolute chill hours read low and the lines read high. Trends are more trustworthy than levels. The elevation profile uses a fixed lapse rate (6.5 °C/km, checked against 5 °C/km), so the chill line's position is modelled while its movement over time comes from the reanalysis. Trend p-values are Mann–Kendall. No open, multi-decade apple production series was found (state and district figures sit in paywalled databases and PDF tables), so this doesn't test effects on yield.</p>
<p class="note">Code: <code>analysis/chill.py</code> in the himalaya-nowcast-atlas repo. Data: ERA5-Land via the Open-Meteo historical archive.</p>
</main></body></html>"""
    with open(os.path.join(HERE, "chill_report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote analysis/chill_report.html")


if __name__ == "__main__":
    main()
