"""Owner: Data Lead.

Single source of truth for the 6 robustness transform families. Used by BOTH
training augmentation (dataset.py) and evaluation (evaluate.py) — do not
duplicate these anywhere else in the repo. Severities here must match
configs/train.yaml's `augmentation` section.

JPEG compression must be a real PIL encode/decode roundtrip, not a
differentiable approximation, to faithfully match the brief's real-world
analog.
"""

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Based on Hackathon's augmentations found in the Information Document
JPEG_QUALITIES = [90, 70, 50, 30]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
RESIZE_SCALES = [0.5, 0.25]
NOISE_SIGMAS = [0.02, 0.05, 0.10]
COLOR_JITTER_PCT = 0.20
CENTER_CROP_FRAC = 0.80


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    original_mode = image.mode
    rgb_image = image if original_mode == "RGB" else image.convert("RGB")
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed.load()
    return compressed if original_mode == "RGB" else compressed.convert(original_mode)


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_then_upscale(image: Image.Image, scale: float) -> Image.Image:
    original_size = image.size
    small_size = (
        max(1, round(original_size[0] * scale)),
        max(1, round(original_size[1] * scale)),
    )
    downscaled = image.resize(small_size, Image.BILINEAR)
    return downscaled.resize(original_size, Image.BILINEAR)


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, array.shape).astype(np.float32)
    noisy = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8))


def color_jitter(image: Image.Image, pct: float = COLOR_JITTER_PCT) -> Image.Image:
    result = image
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = random.uniform(1.0 - pct, 1.0 + pct)
        result = enhancer_cls(result).enhance(factor)
    return result


def center_crop(image: Image.Image, frac: float = CENTER_CROP_FRAC) -> Image.Image:
    original_size = image.size
    crop_w = max(1, round(original_size[0] * frac))
    crop_h = max(1, round(original_size[1] * frac))
    left = (original_size[0] - crop_w) // 2
    top = (original_size[1] - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize(original_size, Image.BILINEAR)


# Canonical transform names — must match configs/train.yaml's `augmentation`
# keys exactly (see docs/interfaces.md #4). evaluate.py should iterate this
# dict, not hardcode names, to build the robustness matrix.
TRANSFORM_SEVERITIES = {
    "jpeg_quality": JPEG_QUALITIES,
    "blur_sigma": BLUR_SIGMAS,
    "resize_scale": RESIZE_SCALES,
    "noise_sigma": NOISE_SIGMAS,
    "color_jitter_pct": [COLOR_JITTER_PCT],
    "center_crop_frac": [CENTER_CROP_FRAC],
}

_DISPATCH = {
    "jpeg_quality": lambda image, severity: jpeg_compress(image, int(severity)),
    "blur_sigma": lambda image, severity: gaussian_blur(image, float(severity)),
    "resize_scale": lambda image, severity: resize_then_upscale(image, float(severity)),
    "noise_sigma": lambda image, severity: gaussian_noise(image, float(severity)),
    "color_jitter_pct": lambda image, severity: color_jitter(image, float(severity)),
    "center_crop_frac": lambda image, severity: center_crop(image, float(severity)),
}


def random_transform_report(image: Image.Image, apply_prob: float) -> tuple[Image.Image, bool]:
    """Like `random_transform`, but also returns whether a transform was
    actually applied. Training uses this to decide whether a sample's cached
    (clean) CLIP embedding is still valid — an augmented image needs a live
    re-embed (see train.py / data/clip_embedding_cache.py)."""
    if random.random() > apply_prob:
        return image.copy(), False
    name = random.choice(list(TRANSFORM_SEVERITIES))
    severity = random.choice(TRANSFORM_SEVERITIES[name])
    return apply_named(image, name, severity), True


def random_transform(image: Image.Image, apply_prob: float) -> Image.Image:
    """Randomly applies one transform family at a random severity, or leaves
    the image clean with probability (1 - apply_prob). Used for train-time
    augmentation."""
    return random_transform_report(image, apply_prob)[0]


def apply_named(image: Image.Image, name: str, severity) -> Image.Image:
    """Deterministic dispatch by name+severity — used by evaluate.py to build
    the robustness matrix (needs a specific transform/severity, not a random
    one)."""
    try:
        func = _DISPATCH[name]
    except KeyError:
        raise ValueError(f"Unknown transform name: {name!r}. Valid names: {list(_DISPATCH)}")
    return func(image, severity)
