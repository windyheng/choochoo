"""Owner: Data Lead.

Downloads/prepares SID_Set (HuggingFace), CIFAKE (Kaggle), and WildFake
(ModelScope — must be manually translated via the site's translation button
before this script can consume it; not automatable).

Responsibilities:
- Download raw data into data/raw/ (gitignored).
- Dedup/near-duplicate check across sources and against each other.
- Build fixed train/val/test split CSVs (image_path, label, source) under
  data/cache/splits/ (gitignored — regenerate via this script, don't commit).
- Check and report class balance per source.

Run: python data/prepare_datasets.py --config configs/train.yaml
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
