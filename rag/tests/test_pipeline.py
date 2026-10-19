"""Tests use SYNTHETIC report text that mimics BEA phrasing; the numbers are made up."""
import json

from pce_rag.ingest import extract_metrics, find_reference_month, find_release_date, ingest
from pce_rag.query import answer, parse_period, retrieve

MODERN = """Personal Income and Outlays, December 2023
January 26, 2024
The PCE price index increased 0.2 percent. Excluding food and energy, the PCE price index increased 0.2 percent (table 5).
From the same month one year ago, the PCE price index for December increased 2.6 percent. Excluding food and energy, \
the PCE price index increased 2.9 percent from one year ago (table 7)."""

OLDER = """Personal Income and Outlays: June 2015
July 30, 2015
The price index for PCE decreased 0.1 percent in June, compared with an increase of 0.3 percent in May. \
The PCE price index, excluding food and energy, was unchanged, in contrast to an increase of 0.1 percent in May.
The price index for PCE increased 0.3 percent from one year ago. The PCE price index, excluding food and energy, \
increased 1.3 percent from one year ago."""


def synthetic(year, month, mom, yoy, extra=""):
    names = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
    verb = "increased" if mom >= 0 else "decreased"
    return (f"Personal Income and Outlays, {names[month - 1]} {year}\n{names[month % 12]} 28, {year + (month == 12)}\n\n"
            f"The PCE price index {verb} {abs(mom):.1f} percent. Excluding food and energy, the PCE price index "
            f"increased 0.2 percent.\n\nFrom the same month one year ago, the PCE price index increased {yoy:.1f} percent. "
            f"Excluding food and energy, the PCE price index increased {yoy - 0.3:.1f} percent from one year ago.\n\n{extra}")


def test_extract_modern_phrasing():
    assert extract_metrics(MODERN) == {"headline_mom": 0.2, "core_mom": 0.2, "headline_yoy": 2.6, "core_yoy": 2.9}


def test_extract_older_phrasing_uses_first_verb_and_signs():
    assert extract_metrics(OLDER) == {"headline_mom": -0.1, "core_mom": 0.0, "headline_yoy": 0.3, "core_yoy": 1.3}


def test_unparseable_value_is_left_missing_not_guessed():
    m = extract_metrics("The PCE price index increased less than 0.1 percent, compared with an increase of 0.4 percent.")
    assert "headline_mom" not in m


def test_metadata():
    assert find_reference_month(MODERN, "x.txt") == "2023-12"
    assert find_reference_month(OLDER, "x.txt") == "2015-06"
    assert find_reference_month("no title here", "pi0319.pdf") == "2019-03"
    assert find_release_date(MODERN) == "2024-01-26"


def test_parse_period():
    avail = ["2015-01", "2024-12"]
    assert parse_period("How did core PCE evolve between 2018 and 2020?", avail) == ("2018-01", "2020-12")
    assert parse_period("What was PCE inflation in March 2021?", avail) == ("2021-03", "2021-03")
    assert parse_period("Trend since 2022?", avail) == ("2022-01", "2024-12")
    assert parse_period("Inflation over the last 3 years", avail) == ("2021-12", "2024-12")
    assert parse_period("When did inflation peak?", avail) is None


def test_end_to_end(tmp_path):
    docs = tmp_path / "reports"
    docs.mkdir()
    yoy = [1.5, 1.6, 1.8, 2.4, 3.9, 4.0, 4.2, 6.6, 6.8, 6.3, 5.0, 3.4]  # peaks in month 9
    for i, v in enumerate(yoy):
        extra = "Energy prices surged on supply disruptions." if i == 8 else "Spending on services rose."
        (docs / f"r{i:02d}.txt").write_text(synthetic(2021 + i // 12, i % 12 + 1, 0.3, v, extra))
    (docs / "r12.html").write_text(f"<html><body><p>{synthetic(2022, 1, -0.1, 3.0)}</p></body></html>")

    index_path = tmp_path / "index.json"
    idx = ingest(docs, index_path)
    assert [r["report_id"] for r in idx["reports"]][-1] == "2022-01"
    assert idx["skipped"] == []
    assert json.loads(index_path.read_text())["reports"][-1]["metrics"]["headline_mom"] == -0.1

    ctx = answer(index_path, "When did headline PCE inflation peak in 2021?", use_llm=False)
    assert "max 6.8 (2021-09)" in ctx
    assert "2022-01 |" not in ctx  # out-of-period report excluded from the table

    hits = retrieve(idx, "energy prices supply disruptions", ("2021-01", "2021-12"))
    assert hits[0]["report_id"] == "2021-09"
    assert all(h["report_id"].startswith("2021") for h in hits)
