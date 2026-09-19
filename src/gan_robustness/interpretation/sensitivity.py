"""Discriminator decision sensitivity: saliency, Grad-CAM and occlusion.

All maps answer "which regions make an image look real to D": saliency and
Grad-CAM via gradients of the realness logit, occlusion via the drop in the
realness probability when a 4x4 window is blanked (per the methodichka LR3).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from gan_robustness.config import IMAGE_SIDE
from gan_robustness.models.discriminator import Discriminator


def saliency(
    discriminator: Discriminator, image: NDArray[np.float32], device: torch.device
) -> NDArray[np.float64]:
    """Return ``|d logit / d x|`` as a 28x28 map for one image."""
    discriminator.eval()
    x = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    logit = discriminator(x).view(-1)
    logit.backward()
    assert x.grad is not None
    return cast("NDArray[np.float64]", x.grad[0, 0].abs().cpu().numpy().astype(np.float64))


def grad_cam(
    discriminator: Discriminator, image: NDArray[np.float32], device: torch.device
) -> NDArray[np.float64]:
    """Return a Grad-CAM map (28x28) from the last discriminator conv block."""
    discriminator.eval()
    x = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
    feats = discriminator.features_net(x)
    feats.requires_grad_(True)
    feats.retain_grad()
    logit = discriminator.classifier(feats).view(-1)
    discriminator.zero_grad()
    logit.backward()
    grads = feats.grad
    assert grads is not None
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feats).sum(dim=1, keepdim=True).clamp(min=0.0)
    cam = F.interpolate(cam, size=(IMAGE_SIDE, IMAGE_SIDE), mode="bilinear", align_corners=False)
    cam_np = cam[0, 0].detach().cpu().numpy().astype(np.float64)
    peak = float(cam_np.max())
    return cast("NDArray[np.float64]", cam_np / peak if peak > 0 else cam_np)


def occlusion(
    discriminator: Discriminator,
    image: NDArray[np.float32],
    device: torch.device,
    kernel: int = 4,
    stride: int = 1,
    fill: float = -1.0,
) -> NDArray[np.float64]:
    """Return the drop in D realness probability under 4x4 occlusion (28x28 map).

    Args:
        discriminator: Trained discriminator.
        image: Image ``(1, 28, 28)`` in ``[-1, 1]``.
        device: Torch device.
        kernel: Occlusion window size.
        stride: Sliding stride.
        fill: Fill value (``-1`` = background on the ``[-1, 1]`` scale).

    Returns:
        A 28x28 map of realness-probability drop.
    """
    discriminator.eval()
    base = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        base_prob = float(torch.sigmoid(discriminator(base)).item())
    positions = [
        (r, c)
        for r in range(0, IMAGE_SIDE - kernel + 1, stride)
        for c in range(0, IMAGE_SIDE - kernel + 1, stride)
    ]
    batch = base.repeat(len(positions), 1, 1, 1)
    for i, (r, c) in enumerate(positions):
        batch[i, 0, r : r + kernel, c : c + kernel] = fill
    with torch.no_grad():
        probs = torch.sigmoid(discriminator(batch)).cpu().numpy()
    drops = base_prob - probs
    buffer = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float64)
    counter = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float64)
    for drop, (r, c) in zip(drops, positions, strict=True):
        buffer[r : r + kernel, c : c + kernel] += drop
        counter[r : r + kernel, c : c + kernel] += 1.0
    counter[counter == 0.0] = 1.0
    return buffer / counter
