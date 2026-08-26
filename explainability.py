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
    raise NotImplementedError
