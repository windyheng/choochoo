"""Owners: CLIP Branch Lead (attention rollout/Grad-CAM section) + SRM Branch
Lead (residual-map visualization section).

Produces the visual overlays for the demo video: CLIP attention rollout /
Grad-CAM over ViT patches, and the raw SRM residual map. Can be built and
sanity-checked against a pretrained/untrained model early — doesn't need to
wait for the final trained checkpoint, just re-run once it exists.
"""


def clip_attention_overlay(image_path: str, checkpoint_path: str):
    """Owner: CLIP Branch Lead."""
    raise NotImplementedError


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
