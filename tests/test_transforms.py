"""Owner: Data Lead. Unit tests for data/transforms.py — the module
train-time augmentation AND evaluate.py both depend on. Keep these passing;
other people's code assumes these contracts hold.
"""

import pytest
from PIL import Image

from data import transforms


@pytest.fixture
def sample_image():
    return Image.new("RGB", (256, 256), color=(128, 64, 200))


@pytest.mark.parametrize("quality", transforms.JPEG_QUALITIES)
def test_jpeg_compress_preserves_size_and_mode(sample_image, quality):
    out = transforms.jpeg_compress(sample_image, quality)
    assert out.size == sample_image.size
    assert out.mode == sample_image.mode


@pytest.mark.parametrize("sigma", transforms.BLUR_SIGMAS)
def test_gaussian_blur_preserves_size(sample_image, sigma):
    out = transforms.gaussian_blur(sample_image, sigma)
    assert out.size == sample_image.size


@pytest.mark.parametrize("scale", transforms.RESIZE_SCALES)
def test_resize_then_upscale_returns_original_size(sample_image, scale):
    out = transforms.resize_then_upscale(sample_image, scale)
    assert out.size == sample_image.size


def test_center_crop_returns_original_size(sample_image):
    out = transforms.center_crop(sample_image)
    assert out.size == sample_image.size
