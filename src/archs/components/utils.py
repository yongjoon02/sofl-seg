import torch
import torch.nn.functional as F
import random

def random_patch_batch(tensors, patch_size, num_patches):
    if patch_size is None or num_patches <= 0:
        return tensors
    ref = tensors[0]
    batch, _, height, width = ref.shape
    if patch_size > height or patch_size > width:
        return tensors
    total = batch * num_patches
    device = ref.device
    batch_indices = torch.arange(batch, device=device).repeat_interleave(num_patches)
    max_top = height - patch_size
    max_left = width - patch_size
    if max_top < 0 or max_left < 0:
        return tensors
    top = torch.randint(0, max_top + 1, (total,), device=device)
    left = torch.randint(0, max_left + 1, (total,), device=device)
    base_y = torch.arange(patch_size, device=device, dtype=torch.float32)
    base_x = torch.arange(patch_size, device=device, dtype=torch.float32)
    grid_y = top.unsqueeze(1) + base_y.unsqueeze(0)
    grid_x = left.unsqueeze(1) + base_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(2).expand(-1, -1, patch_size)
    grid_x = grid_x.unsqueeze(1).expand(-1, patch_size, -1)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    if width > 1:
        grid[..., 0] = (grid[..., 0] / (width - 1)) * 2 - 1
    else:
        grid[..., 0] = 0
    if height > 1:
        grid[..., 1] = (grid[..., 1] / (height - 1)) * 2 - 1
    else:
        grid[..., 1] = 0
    patches = []
    for tensor in tensors:
        expanded = tensor[batch_indices]
        needs_float = not torch.is_floating_point(expanded)
        expanded_f = expanded.float() if needs_float else expanded
        patch = F.grid_sample(expanded_f, grid, mode="bilinear", align_corners=True)
        if needs_float:
            patch = patch.to(expanded.dtype)
        patches.append(patch)
    return patches

def select_patch_params(patch_plan: list = None):
    """Randomly select patch size and number of patches from patch_plan.
    
    Args:
        patch_plan: List of (patch_size, num_patches) tuples
        
    Returns:
        (patch_size, num_patches) tuple
    """
    if patch_plan is None:
        patch_plan = [(320, 6), (384, 4), (416, 3), (512, 1)]
    return random.choice(patch_plan)