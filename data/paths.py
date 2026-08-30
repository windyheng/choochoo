"""Owner: Data Lead.

Single source of truth for resolving split-CSV image paths. Split CSVs
(data/cache/splits/*.csv) and data/raw/sid_set/labels.csv store image_path as
POSIX paths relative to REPO_ROOT, so they're portable across machines and
OSes (a teammate's clone, Colab) rather than hard-wired to whoever generated
them (see docs/interfaces.md §4). Anything that opens an image from a path
sourced from one of those CSVs must resolve it through `resolve_image_path`
first; anything that writes one should normalize through `to_repo_relative`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def to_repo_relative(path) -> str:
    """Normalizes a path for writing to a split CSV: absolute paths under
    REPO_ROOT become REPO_ROOT-relative POSIX strings; anything already
    relative, or absolute but outside REPO_ROOT, is returned as-is (as
    POSIX) so the function is idempotent."""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(REPO_ROOT)
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def resolve_image_path(image_path) -> Path:
    """Resolves a split-CSV image_path for opening. A relative path is
    anchored at REPO_ROOT (independent of the caller's CWD); an absolute
    path (e.g. ad-hoc CLI input outside the splits) passes through
    unchanged."""
    p = Path(image_path)
    return p if p.is_absolute() else REPO_ROOT / p
