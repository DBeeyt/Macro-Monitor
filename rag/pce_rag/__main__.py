"""CLI: python -m pce_rag ingest <dir> | python -m pce_rag ask "<question>" [--no-llm]"""
import argparse
from pathlib import Path

from .ingest import ingest
from .query import answer


def main():
    p = argparse.ArgumentParser(prog="pce_rag")
    p.add_argument("--index", type=Path, default=Path("pce_index.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="parse a folder of PCE releases (.pdf/.html/.txt)")
    ing.add_argument("input_dir", type=Path)
    ask = sub.add_parser("ask", help="answer a question from the index")
    ask.add_argument("question")
    ask.add_argument("--no-llm", action="store_true", help="print the retrieved context instead of calling Claude")
    args = p.parse_args()

    if args.cmd == "ingest":
        idx = ingest(args.input_dir, args.index)
        reports = idx["reports"]
        print(f"Indexed {len(reports)} reports, {len(idx['chunks'])} chunks -> {args.index}")
        if reports:
            print(f"Coverage: {reports[0]['report_id']} to {reports[-1]['report_id']}")
        incomplete = [r["report_id"] for r in reports if len(r["metrics"]) < 4]
        if incomplete:
            print(f"Missing some headline figures: {', '.join(incomplete)}")
        for s in idx["skipped"]:
            print(f"Skipped {s}")
    else:
        print(answer(args.index, args.question, use_llm=not args.no_llm))


if __name__ == "__main__":
    main()
