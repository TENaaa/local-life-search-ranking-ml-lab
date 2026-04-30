from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import write_analysis_outputs
from .config import DEFAULT_DATA_DIR, DEFAULT_DB_PATH, DEFAULT_FIGURE_DIR, DEFAULT_METRICS_PATH, DEFAULT_MODEL_CARD_PATH, DEFAULT_MODEL_DIR, DEFAULT_REPORT_PATH
from .data_generation import generate_synthetic_data
from .database import build_database
from .modeling import train_and_evaluate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="本地生活搜索排序、转化预估与反事实评估")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR)
    generate.add_argument("--users", type=int, default=4500)
    generate.add_argument("--merchants", type=int, default=720)
    generate.add_argument("--queries", type=int, default=240)
    generate.add_argument("--sessions", type=int, default=9000)
    generate.add_argument("--seed", type=int, default=42)

    build = sub.add_parser("build-db")
    build.add_argument("--csv-dir", type=Path, default=DEFAULT_DATA_DIR)
    build.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    train = sub.add_parser("train")
    train.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    train.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    evaluate.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    evaluate.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    report = sub.add_parser("report")
    report.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    report.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    report.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    report.add_argument("--model-card", type=Path, default=DEFAULT_MODEL_CARD_PATH)
    report.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--sessions", type=int, default=9000)
    run_all.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_synthetic_data(args.out, args.users, args.merchants, args.queries, args.sessions, seed=args.seed)
        _print_counts("CSV generated", result.row_counts)
    elif args.command == "build-db":
        _print_counts(f"SQLite built: {args.db}", build_database(args.csv_dir, args.db))
    elif args.command in {"train", "evaluate"}:
        result = train_and_evaluate(args.db, args.model_dir, args.metrics)
        print(f"Metrics: {result.metrics_path}")
        print(f"Predictions: {result.predictions_path}")
    elif args.command == "report":
        counts = write_analysis_outputs(args.db, args.metrics, args.report, args.model_card, args.figures)
        _print_counts(f"Report generated: {args.report}", counts)
    elif args.command == "run-all":
        generated = generate_synthetic_data(DEFAULT_DATA_DIR, n_sessions=args.sessions, seed=args.seed)
        _print_counts("CSV generated", generated.row_counts)
        _print_counts(f"SQLite built: {DEFAULT_DB_PATH}", build_database(DEFAULT_DATA_DIR, DEFAULT_DB_PATH))
        trained = train_and_evaluate(DEFAULT_DB_PATH, DEFAULT_MODEL_DIR, DEFAULT_METRICS_PATH)
        print(f"Metrics: {trained.metrics_path}")
        counts = write_analysis_outputs(DEFAULT_DB_PATH, DEFAULT_METRICS_PATH, DEFAULT_REPORT_PATH, DEFAULT_MODEL_CARD_PATH, DEFAULT_FIGURE_DIR)
        _print_counts(f"Report generated: {DEFAULT_REPORT_PATH}", counts)


def _print_counts(title: str, counts: dict[str, int]) -> None:
    print(title)
    for name, count in counts.items():
        print(f"  - {name}: {count:,}")


if __name__ == "__main__":
    main()
