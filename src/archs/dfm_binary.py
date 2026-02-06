"""Discrete/binary flow matching utilities (DFM/Binary FM)."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def to_binary_mask(x: Tensor) -> Tensor:
    """Convert a mask to {0,1} float tensor without touching global pipelines."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    # Heuristic: if values are in [-1, 1], threshold at 0; else use 0.5.
    if x.min().item() < 0.0:
        return (x > 0).float()
    return (x > 0.5).float()


def make_xt_continuous(flow_matcher, x0: Tensor, x1: Tensor):
    """Wrapper for the existing continuous flow matching path (no behavior change)."""
    return flow_matcher.sample_location_and_conditional_flow(x0, x1)


def make_xt_binary(x1: Tensor, x0: Tensor | None = None, t: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample DFM/Binary FM replacement path.
    
    Each pixel independently chooses x1 with probability t, x0 with probability (1-t).
    This creates a discrete path where xt ∈ {0,1} at training time.

    Returns:
        t: (B,) uniform in [0, 1]
        xt: (B, 1, H, W) binary tensor in {0,1}
        x0: (B, 1, H, W) binary tensor in {0,1}
    """
    if x1.dim() == 3:
        x1 = x1.unsqueeze(1)
    x1 = x1.float()
    if x0 is None:
        x0 = torch.bernoulli(torch.full_like(x1, 0.5))
    else:
        x0 = x0.float()
    if t is None:
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)

    # Discrete replacement: each pixel chooses x1 with probability t
    t_view = t.view(-1, 1, 1, 1)
    mask = torch.rand((x1.shape[0], 1, x1.shape[2], x1.shape[3]), device=x1.device, dtype=x1.dtype) < t_view
    if x1.shape[1] != 1:
        mask = mask.expand(-1, x1.shape[1], -1, -1)
    xt = torch.where(mask, x1, x0)
    return t, xt, x0


def loss_cfm(
    *,
    use_registry_loss: bool,
    loss_fn,
    compute_loss_fn,
    v: Tensor,
    ut: Tensor,
    xt: Tensor,
    geometry: Tensor,
    t: Tensor,
    geometry_pred: Tensor | None = None,
    hard_labels: Tensor | None = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute continuous flow matching loss with optional registry loss."""
    if use_registry_loss:
        loss, loss_dict = loss_fn(
            v, ut, xt, geometry, t=t, geometry_pred=geometry_pred, hard_labels=hard_labels
        )
        return loss, loss_dict
    return compute_loss_fn(v, ut, xt, geometry, t), {}


def _soft_dice_loss(probs: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if probs.dim() == 3:
        probs = probs.unsqueeze(1)
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def loss_dfm_binary(logits: Tensor, x1: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Binary FM loss: BCEWithLogits + soft Dice."""
    if x1.dim() == 3:
        x1 = x1.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, x1)
    probs = torch.sigmoid(logits)
    dice = _soft_dice_loss(probs, x1)
    return bce + dice, {'bce': bce, 'dice': dice}


def _maybe_capture(
    store: List[Tensor] | Dict[int, Tensor] | Dict[int, Dict[str, Tensor]],
    step: int,
    x: Tensor,
    save_steps: Iterable[int] | None,
    p1: Tensor | None = None,
    p_t: Tensor | None = None,
) -> None:
    """Capture intermediate steps.
    
    Args:
        store: Storage for intermediate steps (list or dict)
        step: Current step index
        x: Sampled binary mask
        save_steps: Steps to save (None = save all)
        p1: Model prediction (x1), optional for dual capture
        p_t: Current probability state, for continuous visualization
    """
    if isinstance(store, dict):
        if save_steps is not None and step in save_steps:
            if p1 is not None:
                # Store prediction, sample, and current probability state
                store[step] = {
                    'pred': p1.detach(),
                    'sample': x.detach(),
                    'p_t': p_t.detach() if p_t is not None else None,
                }
            else:
                # Legacy: store only x
                store[step] = x.detach()
        return
    # Legacy list mode
    store.append(x.detach())


def sampler_dfm_euler(
    model_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    cond: Tensor,
    steps: int,
    *,
    eps: float = 1e-6,
    final_sample: bool = True,
):
    """Euler sampler for binary FM probability ODE."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if cond.dim() == 3:
        cond = cond.unsqueeze(1)

    b, _, h, w = cond.shape
    # Start from p=0.5 (maximum entropy)
    p = torch.full((b, 1, h, w), 0.5, device=cond.device, dtype=cond.dtype)
    # Initial state: stochastic Bernoulli sampling (matches reference implementation)
    x = torch.bernoulli(p)

    dt = 1.0 / steps
    last_p1 = None

    for k in range(steps):
        t = torch.full((b,), k / steps, device=cond.device, dtype=cond.dtype)
        # Model input: binary state x (discrete trajectory)
        logits = model_fn(x, t, cond)
        p1 = torch.sigmoid(logits)
        last_p1 = p1
        t_view = t.view(-1, 1, 1, 1)
        # Velocity: (target - current) / remaining_time
        v = (p1 - p) / (1.0 - t_view + eps)
        # Update probability (continuous)
        p = torch.clamp(p + v * dt, 0.0, 1.0)
        # Binarize for next step (discrete trajectory)
        x = torch.bernoulli(p)

    # Return final probability (continuous value in [0,1])
    if final_sample and last_p1 is not None:
        return last_p1
    return p


def sampler_dfm_heun(
    model_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    cond: Tensor,
    steps: int,
    *,
    eps: float = 1e-6,
    final_sample: bool = True,
):
    """Heun (predictor-corrector) sampler for binary FM probability ODE."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if cond.dim() == 3:
        cond = cond.unsqueeze(1)

    b, _, h, w = cond.shape
    # Start from p=0.5 (maximum entropy)
    p = torch.full((b, 1, h, w), 0.5, device=cond.device, dtype=cond.dtype)
    # Initial state: stochastic Bernoulli sampling (matches reference implementation)
    x = torch.bernoulli(p)

    dt = 1.0 / steps
    last_p1 = None

    for k in range(steps):
        t = torch.full((b,), k / steps, device=cond.device, dtype=cond.dtype)
        t_next = torch.full((b,), (k + 1) / steps, device=cond.device, dtype=cond.dtype)

        # Predictor: evaluate velocity at current point
        logits = model_fn(x, t, cond)
        p1 = torch.sigmoid(logits)
        last_p1 = p1
        t_view = t.view(-1, 1, 1, 1)
        v = (p1 - p) / (1.0 - t_view + eps)

        # Predicted next probability (Euler step)
        p_pred = torch.clamp(p + v * dt, 0.0, 1.0)
        x_pred = torch.bernoulli(p_pred)  # Binarize predicted state

        # Corrector: evaluate velocity at predicted point
        logits_next = model_fn(x_pred, t_next, cond)
        p1_next = torch.sigmoid(logits_next)
        t_next_view = t_next.view(-1, 1, 1, 1)
        v_next = (p1_next - p_pred) / (1.0 - t_next_view + eps)

        # Heun update: average of two velocities
        p = torch.clamp(p + 0.5 * (v + v_next) * dt, 0.0, 1.0)
        # Binarize for next step (discrete trajectory)
        x = torch.bernoulli(p)

    # Return final probability (continuous value in [0,1])
    if final_sample and last_p1 is not None:
        return last_p1
    return p
