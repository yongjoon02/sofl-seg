"""Check which decoder stages have FiLM in a checkpoint."""
import sys
import torch

checkpoint_path = sys.argv[1]

# Load checkpoint
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n=== Checkpoint Hyperparameters ===")
if 'hyper_parameters' in ckpt:
    hparams = ckpt['hyper_parameters']
    print(f"use_vlm_film: {hparams.get('use_vlm_film', False)}")
    print(f"vlm_film_decoder_stages: {hparams.get('vlm_film_decoder_stages', 'NOT SET')}")
    print(f"junction_gating_config: {hparams.get('junction_gating_config', {})}")
else:
    print("No hyper_parameters found")

print("\n=== FiLM Weights in Checkpoint ===")
state_dict = ckpt.get('state_dict', {})
film_keys = [k for k in state_dict.keys() if 'vlm_film_heads' in k]
print(f"Found {len(film_keys)} FiLM-related keys")

# Group by stage
stages = set()
for key in film_keys:
    if 'vlm_film_heads.' in key:
        # Extract stage number
        parts = key.split('.')
        stage_idx = int(parts[1])
        stages.add(stage_idx)

print(f"FiLM heads present for stages: {sorted(stages)}")
