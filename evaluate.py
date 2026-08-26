"""Owner: Eval & Deliverables Lead.

Builds the robustness matrix required by the "Robustness Evaluation Summary"
deliverable: clean + every transform/severity in data/transforms.py + at
least one combined transform (e.g. resize->JPEG). Also runs the ablation
(full model vs. CLIP-branch-only vs. artifact-branch-only) per transform.

Metric functions here can be unit-tested independently of a real checkpoint
(tests/test_transforms.py-style) before a trained model exists — build the
harness early against a dummy/random model, then re-run once real
checkpoints land.

Run: python evaluate.py --checkpoint <path> --out results/robustness_table.csv
"""

import argparse


def build_robustness_matrix(checkpoint_path: str):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/robustness_table.csv")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
