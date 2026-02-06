"""MedSegDiff UNet with VLM-FiLM modulation hooks (modified: Stage 4/3 post-concat only)."""

from __future__ import annotations

import torch

from src.archs.components.diffusion_unet import MedSegDiffUNet
from src.archs.components.vlm_film import apply_vlm_film


class MedSegDiffUNetVLMFiLM(MedSegDiffUNet):
    """MedSegDiff UNet with optional VLM-FiLM conditioning in the decoder.
    
    FiLM application strategy:
    - Stage 4 (stage_idx=0): post-concat, pre-blocks
    - Stage 3 (stage_idx=1): post-concat, pre-blocks
    - Stage 2/1: NO FiLM (bypass)
    - Legacy "post-block2, pre-attention" path disabled
    """

    def forward(self, x, time, cond, vlm_cond: dict | None = None, vlm_film_heads=None):
        skip_connect_c = self.skip_connect_condition_fmap

        x = self.init_conv(x)
        c = self.cond_init_conv(cond)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for (block1, block2, attn, dsample), (cond_block1, cond_block2, cond_attn, cond_dsample), conditioner in \
                zip(self.downs, self.cond_downs, self.conditioners):
            x = self._checkpoint(block1, x, t)
            c = self._checkpoint(cond_block1, c, t)
            h.append([x, c] if skip_connect_c else [x])

            x = self._checkpoint(block2, x, t)
            c = self._checkpoint(cond_block2, c, t)
            x = self._checkpoint(attn, x)
            c = self._checkpoint(cond_attn, c)
            x = self._checkpoint(conditioner, x, c)  # FFT conditioning with conditional features
            h.append([x, c] if skip_connect_c else [x])

            x = self._checkpoint(dsample, x)
            c = self._checkpoint(cond_dsample, c)

        x = self._checkpoint(self.mid_block1, x, t)
        c = self._checkpoint(self.cond_mid_block1, c, t)
        x = x + c
        x = self._checkpoint(self.mid_transformer, x, c)
        x = self._checkpoint(self.mid_block2, x, t)

        # Decoder stages: stage_idx mapping to "Layer N"
        # stage_idx=0 → Layer 4 (lowest resolution, first after bottleneck)
        # stage_idx=1 → Layer 3
        # stage_idx=2 → Layer 2
        # stage_idx=3 → Layer 1 (highest resolution)
        for stage_idx, (block1, block2, attn, upsample_layer) in enumerate(self.ups):
            # First skip-connection concatenation
            x = torch.cat((x, *h.pop()), dim=1)
            
            # Process with block1 first to get consistent channel count
            x = self._checkpoint(block1, x, t)
            
            # Apply FiLM based on _vlm_film_stage_indices
            if vlm_cond is not None and vlm_film_heads is not None:
                stage_indices = vlm_cond.get('_vlm_film_stage_indices')
                if not hasattr(self, '_logged_decoder_stages'):
                    print(f"[Decoder] stage_indices from vlm_cond: {stage_indices}")
                    self._logged_decoder_stages = True
                if stage_indices is not None and stage_idx in stage_indices:
                    film_head_idx = stage_indices.index(stage_idx)
                    if not hasattr(self, f'_logged_film_apply_{stage_idx}'):
                        print(f"[Decoder] Applying FiLM at stage_idx={stage_idx}, film_head_idx={film_head_idx}")
                        setattr(self, f'_logged_film_apply_{stage_idx}', True)
                    x, _payload = apply_vlm_film(
                        x,
                        vlm_cond=vlm_cond,
                        vlm_film_heads=vlm_film_heads,
                        stage_idx=film_head_idx,
                        module=self,
                        apply_point="post_block1_pre_concat2",
                    )
            
            # Second skip-connection concatenation
            x = torch.cat((x, *h.pop()), dim=1)
            
            x = self._checkpoint(block2, x, t)
            
            # REMOVED: Legacy FiLM application "post-block2, pre-attention" path
            
            x = self._checkpoint(attn, x)
            x = self._checkpoint(upsample_layer, x)

        x = torch.cat((x, r), dim=1)
        x = self._checkpoint(self.final_res_block, x, t)
        return self.final_conv(x)

