import argparse

from chemteb_retrieval_dataloader.filter_fineweb import run_filter_fineweb


def main() -> None:
    parser = argparse.ArgumentParser(prog="chemfilter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fw = sub.add_parser("filter-fineweb", help="Stream FineWeb-Edu and filter chemistry papers.")
    fw.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    fw.add_argument("--config", default="sample-10BT", help="HF config/name (e.g. sample-10BT, CC-MAIN-2024-10).")
    fw.add_argument("--split", default="train")
    fw.add_argument("--out", required=True, help="Output directory (will be created).")
    fw.add_argument("--max-rows", type=int, default=0, help="Stop after N rows (0 = no limit).")
    fw.add_argument("--max-kept", type=int, default=0, help="Stop after keeping N docs (0 = no limit).")
    fw.add_argument(
        "--keyword-filter",
        action="store_true",
        help="(Deprecated) Keyword filtering is always used now.",
    )
    fw.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Keyword to match (repeatable). If omitted, uses a default chemistry keyword list.",
    )
    fw.add_argument(
        "--min-weak-keyword-hits",
        type=int,
        default=6,
        help="Require at least N weak keyword matches (used when strong threshold is not met).",
    )
    fw.add_argument(
        "--min-strong-keyword-hits",
        type=int,
        default=1,
        help="Require at least N strong keyword matches.",
    )
    fw.add_argument(
        "--url-contains",
        action="append",
        default=[],
        help="If provided, only keep rows whose URL contains this substring (repeatable).",
    )
    fw.add_argument(
        "--url-regex",
        default="",
        help="If provided, only keep rows whose URL matches this regex (case-insensitive).",
    )
    fw.add_argument(
        "--disable-domain-filter",
        action="store_true",
        help="Disable simple domain exclusions (wikipedia/blogs/etc).",
    )
    fw.add_argument("--chunk-tokens", type=int, default=320, help="Approx tokens per chunk (whitespace approx).")
    fw.add_argument("--chunk-overlap", type=int, default=40, help="Approx token overlap per chunk.")
    fw.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing progress in --out (default: true).",
    )

    args = parser.parse_args()

    if args.cmd == "filter-fineweb":
        run_filter_fineweb(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            out_dir=args.out,
            max_rows=args.max_rows,
            max_kept=args.max_kept,
            keywords=args.keyword,
            min_weak_keyword_hits=args.min_weak_keyword_hits,
            min_strong_keyword_hits=args.min_strong_keyword_hits,
            url_contains=args.url_contains,
            url_regex=args.url_regex,
            disable_domain_filter=args.disable_domain_filter,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            resume=args.resume,
        )
        return

    raise SystemExit(f"Unknown command: {args.cmd}")

