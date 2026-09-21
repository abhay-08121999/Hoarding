#!/usr/bin/env python3
"""
One-command build.

    python build.py            clean -> features -> evaluate -> export web data
    python build.py --quick    skip the ablation sweep (much faster)

Everything the web app needs ends up in web/js/data.js.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from src import config
from src.data_loader import build_clean_frame, save_clean
from src.evaluate import ablation, diversity_sweep, evaluate, format_report
from src.export_web import export
from src.features import build_feature_space
from src.recommender import MovieRecommender


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the movie recommender")
    parser.add_argument("--quick", action="store_true",
                        help="skip ablation and diversity sweeps")
    parser.add_argument("--k", type=int, default=10,
                        help="cut-off for evaluation metrics")
    args = parser.parse_args()

    t0 = time.time()

    print("1/5  Loading and cleaning source workbooks")
    df = build_clean_frame()
    save_clean(df)
    print(f"     {df.attrs['rows_read']} rows read -> {len(df)} unique films "
          f"({df.attrs['duplicates_merged']} cross-file duplicates merged)")
    print(f"     saved {config.CLEAN_CSV.name} and {config.CLEAN_JSON.name}")

    print("2/5  Building feature space")
    space = build_feature_space(df)
    print(f"     TF-IDF {space.tfidf.shape[0]}x{space.tfidf.shape[1]}  "
          f"({space.tfidf.nnz} non-zeros)")
    print(f"     categorical {space.categorical.shape[1]} tags, "
          f"people {space.people.shape[1]} names")

    print("3/5  Initialising recommender")
    rec = MovieRecommender(space)

    print(f"4/5  Evaluating (k={args.k})")
    headline = evaluate(rec, k=args.k, mmr_lambda=1.0)
    for key in ["genre_precision@k", "theme_overlap@k", "director_recall@k",
                "intra_list_diversity", "catalogue_coverage"]:
        print(f"     {key:24s} {headline[key]}")

    report_parts = [format_report([headline], f"HEADLINE (k={args.k})")]

    if not args.quick:
        print("     running ablation study")
        abl = ablation(rec, k=args.k)
        report_parts.append(format_report(abl, f"ABLATION (k={args.k})"))
        print("     running MMR diversity sweep")
        sweep = diversity_sweep(rec, k=args.k)
        report_parts.append(format_report(sweep, f"MMR SWEEP (k={args.k})"))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = "\n\n".join(report_parts)
    (config.REPORTS_DIR / "evaluation.txt").write_text(report_text)
    (config.REPORTS_DIR / "evaluation.json").write_text(
        json.dumps(headline, indent=2)
    )
    print(f"     wrote reports/evaluation.txt")

    print("5/5  Exporting web data")
    summary = export(space, rec, metrics=headline)
    print(f"     {summary['path']}  "
          f"({summary['bytes'] / 1024:.0f} KB, {summary['movies']} films, "
          f"{summary['vocab']} terms)")

    print(f"\nDone in {time.time() - t0:.1f}s. "
          f"Open web/index.html, or run: python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
