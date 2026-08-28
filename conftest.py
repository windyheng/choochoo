"""Shared pytest configuration.

`slow` marker: tests that download large model checkpoints (e.g. the CLIP
ViT-B/16 weights in tests/test_backbone_clip.py) or otherwise run too long for
the default / CI pass. Skipped unless `pytest --runslow`.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked `slow`"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (deselected without --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
