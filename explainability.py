"""Owners: CLIP Branch Lead (attention rollout/Grad-CAM section) + SRM Branch
Lead (residual-map visualization section).

Produces the visual overlays for the demo video: CLIP attention rollout /
Grad-CAM over ViT patches, and the raw SRM residual map. Can be built and
sanity-checked against a pretrained/untrained model early — doesn't need to
wait for the final trained checkpoint, just re-run once it exists.
"""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _record_attention(visual):
    """Temporarily makes every ViT block's MultiheadAttention return its
    softmax weights, and collects them. open_clip calls `self.attn(...,
    need_weights=False)` and discards the weights — a forward-pre-hook flips
    that flag, a forward-hook stashes the result. Both are removed on exit."""
    import torch  # noqa: F401  (kept local; module import stays cheap)

    weights = []
    handles = []

    def pre_hook(_module, args, kwargs):
        kwargs = {**kwargs, "need_weights": True, "average_attn_weights": True}
        return args, kwargs

    def post_hook(_module, _args, _kwargs, output):
        # output == (attn_output, attn_output_weights); weights are [B, S, S]
        attn_weights = output[1]
        if attn_weights is not None:
            weights.append(attn_weights.detach())

    for block in visual.transformer.resblocks:
        handles.append(block.attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(block.attn.register_forward_hook(post_hook, with_kwargs=True))
    try:
        yield weights
    finally:
        for h in handles:
            h.remove()


def _attention_rollout(layer_weights):
    """Attention rollout (Abnar & Zuidema, 2020): compose the per-layer
    attention matrices, accounting for the residual connection, and read off
    how much the CLS token attends to each patch.

    layer_weights: list of [1, S, S] tensors (one per transformer block).
    Returns a [grid, grid] tensor in [0, 1].
    """
    import torch

    if not layer_weights:
        raise RuntimeError("no attention weights captured — hook setup failed")

    seq = layer_weights[0].shape[-1]
    eye = torch.eye(seq)
    rollout = eye
    for w in layer_weights:
        a = w[0].to(torch.float32)          # [S, S]
        a = 0.5 * a + 0.5 * eye             # residual connection
        a = a / a.sum(dim=-1, keepdim=True)  # renormalize rows
        rollout = a @ rollout

    cls_to_patches = rollout[0, 1:]          # drop CLS->CLS
    grid = int(round((seq - 1) ** 0.5))
    if grid * grid != seq - 1:
        raise RuntimeError(f"{seq - 1} patch tokens is not a square grid")
    mask = cls_to_patches.reshape(grid, grid)
    mask = mask - mask.min()
    peak = mask.max()
    return mask / peak if peak > 0 else mask


def _heatmap_image(mask, size):
    """[grid, grid] in [0, 1] -> PIL RGB heatmap at `size`, jet colormap."""
    import matplotlib
    import numpy as np
    import torch
    from PIL import Image

    up = torch.nn.functional.interpolate(
        mask[None, None], size=size, mode="bilinear", align_corners=False
    )[0, 0].clamp(0, 1).numpy()
    rgb = (matplotlib.colormaps["jet"](up)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def clip_attention_overlay(
    image_path: str,
    checkpoint_path: str | None = None,
    *,
    save_path: str | None = None,
    alpha: float = 0.5,
):
    """Attention-rollout heatmap for the frozen CLIP ViT, blended over the
    input image. Shows which regions drive CLIP's representation of the image
    — the CLIP branch's contribution to the demo-video overlay.

    `checkpoint_path` is accepted for signature parity with the other
    explainability entry points but is IGNORED: the CLIP backbone is frozen,
    so a trained checkpoint (fusion-head weights only) cannot change its
    attention. Rollout needs no gradients and no trained model.

    Returns a PIL.Image (224x224 RGB). If `save_path` is given, also writes it.
    """
    import numpy as np
    import torch
    from PIL import Image

    from models.backbone_clip import CLIPBackbone

    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)

    backbone = CLIPBackbone()
    with _record_attention(backbone.visual) as layer_weights:
        backbone.embed(x)

    mask = _attention_rollout(layer_weights)
    heatmap = _heatmap_image(mask, size=image.size[::-1])
    overlay = Image.blend(image, heatmap, alpha)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        overlay.save(save_path)
    return overlay


def srm_residual_overlay(image_path: str):
    """Owner: SRM Branch Lead. Doesn't need a trained checkpoint — visualizes
    the fixed SRM filter output directly."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    from models.artifact_branch import ArtifactBranch

    image = Image.open(image_path).convert("RGB")
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0)

    branch = ArtifactBranch()
    with torch.no_grad():
        residuals = F.conv2d(pixels, branch.srm_kernels, padding=1, groups=3)
        intensity = residuals.abs().mean(dim=1, keepdim=True)

    intensity = intensity.squeeze().numpy()
    intensity -= intensity.min()
    maximum = intensity.max()
    if maximum > 0:
        intensity /= maximum

    heatmap = np.zeros((*intensity.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (255 * intensity).astype(np.uint8)
    heatmap[..., 1] = (180 * (1.0 - intensity)).astype(np.uint8)
    heatmap[..., 2] = (255 * (1.0 - intensity)).astype(np.uint8)
    return Image.fromarray(heatmap, mode="RGB")
