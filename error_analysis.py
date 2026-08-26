"""Owner: Eval & Deliverables Lead.

Mines representative false positives/negatives from evaluate.py's results,
bucketed by transform x severity, and saves thumbnail grids under
results/error_analysis/. Feeds the required "Error Analysis Note"
deliverable — pair each bucket with the false-positive-cost trade-off
discussion (see docs/specs design doc, section 7).

Run: python error_analysis.py --predictions results/robustness_table.csv --out results/error_analysis/
"""

import argparse


def mine_errors(predictions_path: str, out_dir: str):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="results/error_analysis/")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
