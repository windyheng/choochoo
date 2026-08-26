"""Owner: Training/Infra Lead.

Training loop with checkpoint/resume — REQUIRED for free-tier Colab, which
disconnects at the ~12hr session cap and sometimes earlier. Must detect and
resume from the latest checkpoint under configs' checkpoint_dir on restart,
not assume a single uninterrupted run. Build this before the real training
run, not after a disconnect eats progress.

Run: python train.py --config configs/train.yaml
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
