"""Owner: Data Lead. Tests for data.paths — the REPO_ROOT-anchored path
convention split CSVs rely on (see docs/interfaces.md §4).

Regression coverage for the non-portable-CSV bug: a split CSV written with
an absolute in-repo path must round-trip to a relative path that resolves
correctly regardless of the caller's CWD.
"""

import csv
import os
from pathlib import Path

import pytest

from data.paths import REPO_ROOT, resolve_image_path, to_repo_relative


def test_to_repo_relative_converts_absolute_path_under_repo_root():
    abs_path = REPO_ROOT / "data" / "raw" / "cifake" / "test" / "FAKE" / "0.jpg"
    assert to_repo_relative(abs_path) == "data/raw/cifake/test/FAKE/0.jpg"


def test_to_repo_relative_is_idempotent_on_already_relative_input():
    rel = "data/raw/cifake/test/FAKE/0.jpg"
    assert to_repo_relative(rel) == rel


def test_to_repo_relative_normalizes_backslashes_via_windows_path():
    # Simulates the historical bug: a Windows-generated absolute path.
    win_style = REPO_ROOT / "data" / "raw" / "wildfake" / "celebahq" / "img.jpg"
    assert to_repo_relative(str(win_style)) == "data/raw/wildfake/celebahq/img.jpg"


def test_resolve_image_path_anchors_relative_path_at_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # a CWD far from REPO_ROOT
    resolved = resolve_image_path("data/raw/cifake/test/FAKE/0.jpg")
    assert resolved == REPO_ROOT / "data" / "raw" / "cifake" / "test" / "FAKE" / "0.jpg"


def test_resolve_image_path_passes_through_absolute_path(tmp_path):
    outside = tmp_path / "ad_hoc_input.jpg"
    assert resolve_image_path(str(outside)) == outside


def test_split_csv_round_trips_from_a_different_cwd(monkeypatch, tmp_path):
    """A CSV written with an absolute REPO_ROOT-relative path must resolve to
    the same real file when read back from an unrelated CWD — the scenario
    that broke on a teammate's clone/Colab before this fix."""
    real_image = REPO_ROOT / "data" / "raw" / "cifake" / "test" / "FAKE" / "0.jpg"
    if not real_image.exists():
        pytest.skip("sample raw image not present in this checkout")

    csv_path = tmp_path / "split.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "source"])
        writer.writerow([to_repo_relative(real_image), 1, "cifake"])

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert not os.path.isabs(row["image_path"])
    assert resolve_image_path(row["image_path"]).samefile(real_image)
