# chemteb-retrieval-dataloader

Stream FineWeb-Edu and **keyword-filter chemistry docs**, then write:
- `docs.jsonl` (kept docs)
- `chunks.jsonl` (sentence-complete chunks for retrieval)

## Setup

```bash
uv sync
```

## Run a small test (20k rows)

```bash
uv run chemfilter filter-fineweb --out "data/filtered/kw_20k" --config sample-10BT --max-rows 20000
```

## Run “full” stream (resume-safe)

```bash
uv run chemfilter filter-fineweb --out "data/filtered/kw_sample-10BT_full" --config sample-10BT
```

Re-run the same command to **resume** (default). Use `--no-resume` to restart from scratch.

## Useful knobs

```bash
# Make filtering stricter/looser
uv run chemfilter filter-fineweb --out "data/filtered/kw_20k" --config sample-10BT --max-rows 20000 --min-keyword-hits 2 --min-strong-keyword-hits 1

# Chunk sizing
uv run chemfilter filter-fineweb --out "data/filtered/kw_20k" --config sample-10BT --max-rows 20000 --chunk-tokens 320 --chunk-overlap 40
```

