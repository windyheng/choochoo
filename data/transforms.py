"""Owner: Data Lead.

Single source of truth for the 6 robustness transform families. Used by BOTH
training augmentation (dataset.py) and evaluation (evaluate.py) — do not
duplicate these anywhere else in the repo. Severities here must match
configs/train.yaml's `augmentation` section.

JPEG compression must be a real PIL encode/decode roundtrip, not a
differentiable approximation, to faithfully match the brief's real-world
analog.
"""

from PIL import Image

JPEG_QUALITIES = [90, 70, 50, 30]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
RESIZE_SCALES = [0.5, 0.25]
NOISE_SIGMAS = [0.02, 0.05, 0.10]
COLOR_JITTER_PCT = 0.20
CENTER_CROP_FRAC = 0.80


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    raise NotImplementedError


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    raise NotImplementedError


def resize_then_upscale(image: Image.Image, scale: float) -> Image.Image:
    raise NotImplementedError


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    raise NotImplementedError


def color_jitter(image: Image.Image, pct: float = COLOR_JITTER_PCT) -> Image.Image:
    raise NotImplementedError


def center_crop(image: Image.Image, frac: float = CENTER_CROP_FRAC) -> Image.Image:
    raise NotImplementedError


def random_transform(image: Image.Image, apply_prob: float) -> Image.Image:
    """Randomly applies one transform family at a random severity, or leaves
    the image clean with probability (1 - apply_prob). Used for train-time
    augmentation."""
    raise NotImplementedError


def apply_named(image: Image.Image, name: str, severity) -> Image.Image:
    """Deterministic dispatch by name+severity — used by evaluate.py to build
    the robustness matrix (needs a specific transform/severity, not a random
    one)."""
    raise NotImplementedError
