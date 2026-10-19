"""Load PCE reports, extract headline inflation figures, chunk text, and build the index."""
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTH_NAMES = r"(January|February|March|April|May|June|July|August|September|October|November|December)"

METRICS = ("headline_mom", "headline_yoy", "core_mom", "core_yoy")


@dataclass
class Report:
    report_id: str            # reference month, "YYYY-MM"
    source: str               # file name
    release_date: str | None  # "YYYY-MM-DD"
    metrics: dict = field(default_factory=dict)  # METRICS -> percent (first print)
    text: str = ""


# ---------- loading ----------

def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text("\n")
    return path.read_text(errors="ignore")


def normalize(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)      # de-hyphenate PDF line breaks
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------- metadata ----------

def find_reference_month(text: str, filename: str) -> str | None:
    """Month the report covers, from the release title or a BEA-style file name (pi1223.pdf)."""
    m = re.search(r"Personal Income and Outlays[,:\s\-–]+" + MONTH_NAMES + r",? (\d{4})", text, re.I)
    if m:
        return f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}"
    m = re.search(r"pi(\d{2})(\d{2})", filename, re.I)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"20{m.group(2)}-{int(m.group(1)):02d}"
    m = re.search(r"(\d{4})[-_](\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def find_release_date(text: str) -> str | None:
    m = re.search(MONTH_NAMES + r" (\d{1,2}), (\d{4})", text[:3000])
    if not m:
        return None
    return f"{int(m.group(3)):04d}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"


# ---------- headline figure extraction ----------

_PCE_PRICE = re.compile(r"PCE price index|price index for PCE", re.I)
_YOY = re.compile(r"one year ago|from the same month|12[- ]month|from a year ago", re.I)
_CORE = re.compile(r"excluding food and energy", re.I)
_VERB = re.compile(r"\b(increased|decreased|rose|fell|declined|was unchanged)\b", re.I)
_VALUE = re.compile(r"^\s*(\d+(?:\.\d+)?) percent")


def _first_change(sentence: str) -> float | None:
    """Value of the first change verb in the sentence; None when it isn't a plain 'X percent'.

    Only the first verb counts: later ones refer to other months ("compared with an increase
    of 0.1 percent in November"). Phrases like "less than 0.1 percent" are left as None
    rather than guessed.
    """
    m = _VERB.search(sentence)
    if not m:
        return None
    verb = m.group(1).lower()
    if verb == "was unchanged":
        return 0.0
    v = _VALUE.match(sentence[m.end():])
    if not v:
        return None
    value = float(v.group(1))
    return -value if verb in ("decreased", "fell", "declined") else value


def extract_metrics(text: str) -> dict:
    """Headline/core, month-over-month/year-over-year PCE price changes as first reported."""
    flat = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(])", flat)
    found = {}
    for s in sentences:
        if not _PCE_PRICE.search(s):
            continue
        key = ("core" if _CORE.search(s) else "headline") + ("_yoy" if _YOY.search(s) else "_mom")
        if key in found:
            continue  # first mention is the headline paragraph
        value = _first_change(s)
        if value is not None:
            found[key] = value
    return found


# ---------- chunking ----------

def chunk_report(report: Report, max_chars: int = 900, overlap: int = 150) -> list[dict]:
    header = f"[PCE report {report.report_id}, released {report.release_date or 'unknown'}] "
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", report.text) if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p)
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(buf)
            buf = buf[-overlap:]
        buf = f"{buf} {p}".strip()
        while len(buf) > max_chars:  # a single very long paragraph
            chunks.append(buf[:max_chars])
            buf = buf[max_chars - overlap:]
    if buf:
        chunks.append(buf)
    return [{"report_id": report.report_id, "chunk_id": f"{report.report_id}#{i}", "text": header + c}
            for i, c in enumerate(chunks)]


# ---------- pipeline ----------

def ingest(input_dir: Path, index_path: Path) -> dict:
    reports, skipped = {}, []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in (".pdf", ".html", ".htm", ".txt"):
            continue
        text = normalize(load_text(path))
        ref = find_reference_month(text, path.name)
        if not ref:
            skipped.append(f"{path.name}: reference month not found")
            continue
        report = Report(ref, path.name, find_release_date(text), extract_metrics(text), text)
        if ref in reports:
            skipped.append(f"{path.name}: duplicate of {reports[ref].source} for {ref}")
            continue
        reports[ref] = report

    chunks = [c for r in reports.values() for c in chunk_report(r)]
    index = {
        "reports": [{k: v for k, v in asdict(r).items() if k != "text"} for r in sorted(reports.values(), key=lambda r: r.report_id)],
        "chunks": chunks,
        "skipped": skipped,
    }
    index_path.write_text(json.dumps(index, indent=1))
    return index
