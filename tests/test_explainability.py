"""Owner: CLIP Branch Lead. Tests for explainability.clip_attention_overlay.

Marked `slow` — loads the CLIP ViT-B/16 checkpoint.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("open_clip")
pytest.importorskip("matplotlib")
from PIL import Image

import explainability

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    # a non-uniform image so attention has something to vary over
    p = tmp_path_factory.mktemp("img") / "x.png"
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[40:150, 60:180] = (200, 120, 60)  # an off-centre block
    Image.fromarray(arr).save(p)
    return str(p)


def test_returns_224_rgb_image(image_path):
    out = explainability.clip_attention_overlay(image_path)
    assert isinstance(out, Image.Image)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_attention_is_not_uniform(image_path):
    # rollout mask should actually vary across patches, not be flat
    import explainability as ex

    from models.backbone_clip import CLIPBackbone

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None]
    bb = CLIPBackbone()
    with ex._record_attention(bb.visual) as w:
        bb.embed(x)
    assert len(w) == 12  # one per ViT-B/16 block
    mask = ex._attention_rollout(w)
    assert mask.shape == (14, 14)
    assert mask.std() > 0.01


def test_deterministic(image_path):
    a = np.asarray(explainability.clip_attention_overlay(image_path))
    b = np.asarray(explainability.clip_attention_overlay(image_path))
    assert np.array_equal(a, b)


def test_hooks_are_removed_after_call(image_path):
    from models.backbone_clip import CLIPBackbone

    bb = CLIPBackbone()
    before = [len(blk.attn._forward_pre_hooks) for blk in bb.visual.transformer.resblocks]
    with explainability._record_attention(bb.visual):
        bb.embed(torch.zeros(1, 3, 224, 224))
    after = [len(blk.attn._forward_pre_hooks) for blk in bb.visual.transformer.resblocks]
    assert before == after  # context manager cleaned up


def test_save_path_writes_file(image_path, tmp_path):
    dest = tmp_path / "sub" / "overlay.png"
    out = explainability.clip_attention_overlay(image_path, save_path=str(dest))
    assert dest.exists()
    assert np.array_equal(np.asarray(Image.open(dest)), np.asarray(out))


def test_checkpoint_path_is_ignored(image_path):
    # passing a bogus checkpoint path must not matter (backbone is frozen)
    a = np.asarray(explainability.clip_attention_overlay(image_path))
    b = np.asarray(explainability.clip_attention_overlay(image_path, checkpoint_path="/no/such.pt"))
    assert np.array_equal(a, b)
