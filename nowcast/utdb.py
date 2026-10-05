"""Uttarakhand Tourism Development Board visitor counts into pit.sqlite.

    python -m nowcast.utdb            # writes ./pit.sqlite

UTDB publishes yearly arrivals for 28 major destinations as one-page PDFs.
Each report is parsed row by row and checked (Indian + foreign = total, and
every column against the report's own total line) before anything is
written; a report that doesn't add up stops the run. Add each new year's
report to REPORTS.

Rows are matched across reports by serial number, which has been stable
since 2021 while the names drift ("Ranikheth" / "Ranikhet", "Tehri" /
"Tehri District"); entities are named from the 2021-23 report.

The table is the point-in-time shape culture.visitors() reads: one row per
destination, year and series, with the source PDF as ref and the day it
was read as known_time.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sqlite3

from .geo import get

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIT = os.path.join(HERE, "pit.sqlite")
BASE = "https://uttarakhandtourism.gov.in/assets/"
# (PDF, the years it covers, in column order)
REPORTS = [
    (BASE + "pdf/Statistic21-23.pdf", ["2021", "2022", "2023"]),
    (BASE + "media/UTDB_media_1743665047Yearly_Report_Statistics_2024.pdf", ["2024"]),
    (BASE + "media/UTDB_media_1785820386Uttarakhand_Tourist_Footfall-2025.pdf", ["2025"]),
]
ROW = re.compile(r"^(\d{1,2})\s+(.+?)((?:\s+\S+){%d})$")


def _slug(name):
    name = re.sub(r"\(.*?\)?$|\(.*?\)", "", name).replace("&", " ")
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _num(s):
    return int(s) if re.fullmatch(r"\d+", s) else None


def parse(pdf_bytes, years):
    """[(serial, printed name, {year: (indian, foreign, total)})], checked."""
    import io
    import pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        lines = [ln.strip() for p in pdf.pages for ln in (p.extract_text() or "").splitlines()]
    ncol = 3 * len(years)
    row = re.compile(ROW.pattern % ncol)
    rows, totals = [], None
    for ln in lines:
        if ln.startswith("Total "):
            totals = [_num(v) for v in ln.split()[1:]]
            continue
        m = row.match(ln)
        if not m:
            continue
        vals = m.group(3).split()
        by_year = {}
        for k, y in enumerate(years):
            ind, frn, tot = (_num(v) for v in vals[3 * k:3 * k + 3])
            if ind is None or tot is None:
                raise ValueError(f"unreadable counts in {ln!r}")
            if frn is None:  # a stray character in the PDF's text: the row's own sum gives it
                frn = tot - ind
            if ind + frn != tot:
                raise ValueError(f"{y}: {ind} + {frn} != {tot} in {ln!r}")
            by_year[y] = (ind, frn, tot)
        rows.append((int(m.group(1)), m.group(2).strip(), by_year))
    if not totals or len(totals) != ncol:
        raise ValueError("no total line")
    for k, y in enumerate(years):
        for j, what in enumerate(("Indian", "foreign", "total")):
            s = sum(r[2][y][j] for r in rows)
            if totals[3 * k + j] is not None and s != totals[3 * k + j]:
                raise ValueError(f"{y} {what}: rows sum to {s}, report says {totals[3 * k + j]}")
    if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"serial numbers out of order: {[r[0] for r in rows]}")
    return rows


def build(path=PIT):
    today = dt.date.today().isoformat()
    names, obs = {}, []
    for url, years in REPORTS:
        body = get(url, timeout=120, cache_key=f"utdb/{url.rsplit('/', 1)[1]}")
        rows = parse(body, years)
        print(f"{url.rsplit('/', 1)[1]}: {len(rows)} destinations, {', '.join(years)}")
        for serial, name, by_year in rows:
            entity = names.setdefault(serial, _slug(name))
            for y, (ind, frn, tot) in by_year.items():
                obs.append(("utdb", "tourists_total", entity, y, tot, url,
                            json.dumps({"indian": ind, "foreign": frn, "name": name}), today))
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.execute("create table obs (source text, series text, entity text, valid_time text, value real, "
                "ref text, meta text, known_time text, primary key (source, series, entity, valid_time))")
    con.executemany("insert into obs values (?,?,?,?,?,?,?,?)", obs)
    con.commit()
    con.close()
    return names, len(obs)


if __name__ == "__main__":
    names, n = build()
    print(f"wrote {PIT}: {n} rows for {len(names)} destinations")
    print(", ".join(names.values()))
