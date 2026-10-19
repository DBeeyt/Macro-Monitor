# PCE inflation RAG (prototype)

Answers questions about historical PCE inflation using BEA "Personal Income and Outlays" releases.

## How it works

1. **Ingest** (`pce_rag/ingest.py`): reads `.pdf` / `.html` / `.txt` releases, detects the reference
   month and release date, pulls out the four headline figures (headline/core, month-over-month/year-over-year),
   and splits the text into chunks tagged with the report month.
2. **Retrieve** (`pce_rag/query.py`): reads any period in the question ("2018 to 2020", "March 2021",
   "since 2022", "last 5 years"), keeps only chunks from that period, and ranks them with BM25.
3. **Ground the numbers**: builds a monthly table plus computed statistics (first/last/min/max/mean)
   for that period, so Claude reads trend numbers from the table instead of working them out from text.
4. **Answer**: Claude (`claude-opus-5-5`) answers from that context only and cites report months.

Why the extra table? About 120 monthly releases use nearly the same wording, so text search alone
can't reliably find "the peak" or "the trend". The table guarantees each covered month is in the context.

## Usage

```bash
pip install -r requirements.txt
# put a decade of releases in ./reports  (e.g. BEA's pi1223.pdf files)
python -m pce_rag ingest ./reports            # writes pce_index.json, lists any reports it couldn't parse
python -m pce_rag ask "How did core PCE inflation trend from 2019 to 2023?"
python -m pce_rag ask "When did headline PCE peak?" --no-llm   # show the retrieved context only
python -m pytest -q tests
```

Claude credentials come from `ANTHROPIC_API_KEY` (or an `ant auth login` profile).

## Known limits

- Figures are **as first reported** in each release. BEA revises them later, so they can differ from
  today's FRED/BEA series. To answer with revised figures, add a revised series (e.g. a FRED `PCEPI` export).
- The figures are pulled out with regexes written for BEA's headline wording. Anything they can't read is
  left blank and reported at ingest, never guessed. Check the ingest output on the real archive.
- Keyword search (BM25) only, no embeddings. That is fine for the narrative text, which uses fixed wording.
  Add embeddings if questions are often phrased very differently from the release text.
