"""Answer questions over the index: period filtering, BM25 retrieval, computed trend facts, Claude synthesis."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from .ingest import METRICS, MONTH_NAMES, MONTHS

METRIC_LABELS = {
    "headline_mom": "PCE price index, % change from prior month",
    "headline_yoy": "PCE price index, % change from year ago",
    "core_mom": "Core PCE (ex food & energy), % change from prior month",
    "core_yoy": "Core PCE (ex food & energy), % change from year ago",
}

_STOP = set("the a an of and or in on for to from is was were what how did does with by at as be it this that "
            "which when over during between since".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def score(self, query: str, i: int) -> float:
        tf, s = self.tfs[i], 0.0
        for t in set(tokenize(query)):
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return s


# ---------- query understanding ----------

def parse_period(question: str, available: list[str]) -> tuple[str, str] | None:
    """Inclusive ("YYYY-MM", "YYYY-MM") range named in the question, or None if none is named."""
    q = question.lower()
    months = [f"{int(y):04d}-{MONTHS[m.lower()]:02d}" for m, y in re.findall(MONTH_NAMES + r",? (\d{4})", question, re.I)]
    years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", re.sub(MONTH_NAMES + r",? \d{4}", "", question, flags=re.I))]
    latest = available[-1] if available else None

    m = re.search(r"(?:last|past) (\d+|two|three|five|ten) years", q)
    if m and latest:
        n = {"two": 2, "three": 3, "five": 5, "ten": 10}.get(m.group(1)) or int(m.group(1))
        y, mo = map(int, latest.split("-"))
        return f"{y - n:04d}-{mo:02d}", latest
    if re.search(r"\b(since|after)\b", q) and (months or years) and latest:
        start = min(months) if months else f"{min(years):04d}-01"
        return start, latest
    bounds = months + [f"{y:04d}-01" for y in years] + [f"{y:04d}-12" for y in years]
    if not bounds:
        return None
    return min(bounds), max(bounds)


# ---------- context assembly ----------

def trend_facts(reports: list[dict]) -> str:
    """Monthly table plus deterministic summary stats, so numbers never come from the LLM's arithmetic."""
    if not reports:
        return "No reports in the index cover this period."
    lines = ["Monthly figures as first reported (report_id = reference month):",
             "report_id | released | " + " | ".join(METRICS)]
    for r in reports:
        vals = [f"{r['metrics'][k]:.1f}" if k in r["metrics"] else "n/a" for k in METRICS]
        lines.append(f"{r['report_id']} | {r['release_date'] or '?'} | " + " | ".join(vals))
    lines.append("\nSummary statistics over this period (computed, not estimated):")
    for k in METRICS:
        pts = [(r["report_id"], r["metrics"][k]) for r in reports if k in r["metrics"]]
        if not pts:
            continue
        lo, hi = min(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[1])
        mean = sum(v for _, v in pts) / len(pts)
        lines.append(f"- {METRIC_LABELS[k]}: first {pts[0][1]:.1f} ({pts[0][0]}), last {pts[-1][1]:.1f} ({pts[-1][0]}), "
                     f"min {lo[1]:.1f} ({lo[0]}), max {hi[1]:.1f} ({hi[0]}), mean {mean:.2f}, n={len(pts)}")
    missing = [r["report_id"] for r in reports if len(r["metrics"]) < len(METRICS)]
    if missing:
        lines.append(f"- Some figures could not be extracted for: {', '.join(missing)}")
    return "\n".join(lines)


def retrieve(index: dict, question: str, period: tuple[str, str] | None, k: int = 8, per_report: int = 2) -> list[dict]:
    chunks = [c for c in index["chunks"] if not period or period[0] <= c["report_id"] <= period[1]]
    if not chunks:
        return []
    bm25 = BM25([c["text"] for c in chunks])
    ranked = sorted(range(len(chunks)), key=lambda i: bm25.score(question, i), reverse=True)
    out, per = [], Counter()
    for i in ranked:  # spread hits across reports instead of piling onto one
        c = chunks[i]
        if per[c["report_id"]] < per_report:
            out.append(c)
            per[c["report_id"]] += 1
        if len(out) == k:
            break
    return out


SYSTEM = """You answer questions about U.S. PCE inflation using only the BEA "Personal Income and Outlays" \
report data supplied in the user's message.

- Take every number from the COMPUTED FACTS table or the quoted passages; never estimate or recall figures \
from memory. If the data needed is missing, say what is missing.
- Cite the report_id(s) (YYYY-MM) behind each figure.
- Figures are as first reported in each month's release; later releases revise them. Say so when it matters.
- For trend questions, describe direction, turning points, and peaks/troughs with their months."""


def build_prompt(question: str, period, facts: str, passages: list[dict]) -> str:
    scope = f"{period[0]} to {period[1]}" if period else "all indexed reports"
    quoted = "\n\n".join(f"<passage report_id=\"{p['report_id']}\">\n{p['text']}\n</passage>" for p in passages)
    return (f"<computed_facts scope=\"{scope}\">\n{facts}\n</computed_facts>\n\n"
            f"<passages>\n{quoted or '(none)'}\n</passages>\n\nQuestion: {question}")


def answer(index_path: Path, question: str, use_llm: bool = True, model: str = "claude-opus-5-5") -> str:
    index = json.loads(index_path.read_text())
    reports = index["reports"]
    period = parse_period(question, [r["report_id"] for r in reports])
    in_scope = [r for r in reports if not period or period[0] <= r["report_id"] <= period[1]]
    facts = trend_facts(in_scope)
    passages = retrieve(index, question, period)
    prompt = build_prompt(question, period, facts, passages)
    if not use_llm:
        return prompt

    import anthropic
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        return "The model declined to answer. Retrieved context:\n\n" + prompt
    return "".join(b.text for b in response.content if b.type == "text")
