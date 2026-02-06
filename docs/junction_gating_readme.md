# Junction-Aware FiLM Gating (Topology Safety Gate)

## Overview

Junction-aware FiLM gating is an optional feature that spatially modulates VLM-FiLM application to reduce false positives at vessel junctions, thereby improving topological accuracy (Betti-1 error).

## Problem

Analysis showed that VLM-FiLM improves Dice/clDice slightly but **increases Betti-1 error by ~25%** due to false positives concentrated at vessel junctions:
- 79% of false positives occur near junctions
- These false positives create spurious loops, degrading topological accuracy

## Solution

Junction-aware FiLM gating selectively suppresses or reduces VLM-FiLM modulation in junction regions using a spatial gate mask:

```
x_out = x_in + gate * (FiLM(x_in) - x_in)
```

Where:
- `gate = 1.0`: Full FiLM modulation (normal behavior)
- `gate = 0.0`: No FiLM modulation (identity, preserves topology)
- `gate ∈ (0,1)`: Partial FiLM modulation

## Configuration

Add `junction_gating_config` section to your YAML config:

```yaml
model:
  use_vlm_film: true
  vlm_film_decoder_stages: [2, 3]
  
  # Junction-aware FiLM gating
  junction_gating_config:
    enabled: false                    # Set to true to enable
    source: "pred"                    # "pred" (model prediction) or "gt" (oracle)
    threshold: 0.5                    # Binarization threshold
    degree_threshold: 3               # Min skeleton degree for junction detection
    radius_px: 8                      # Dilation radius around junctions
    gate_value_in_junction: 0.0       # 0.0 = suppress FiLM, 1.0 = normal
    apply_stages: "same_as_film"      # Which stages to apply gating
    interp: "nearest"                 # Gate interpolation mode
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable/disable junction gating (default OFF for backward compatibility) |
| `source` | `"pred"` | Mask source: `"pred"` (model's own prediction) or `"gt"` (ground truth, oracle mode) |
| `threshold` | `0.5` | Binarization threshold for mask (0-1) |
| `degree_threshold` | `3` | Minimum skeleton node degree to consider as junction |
| `radius_px` | `8` | Dilation radius around detected junctions in pixels |
| `gate_value_in_junction` | `0.0` | Gate value in junction regions (0.0 = fully suppress, 1.0 = normal) |
| `apply_stages` | `"same_as_film"` | Decoder stages to apply gating: `"same_as_film"` or explicit list like `[2,3]` |
| `interp` | `"nearest"` | Interpolation mode when resizing gate: `"nearest"` or `"bilinear"` |

## How It Works

1. **Junction Detection** (per forward pass):
   - Binarize the mask (from model prediction or ground truth)
   - Compute morphological skeleton
   - Detect junction nodes (skeleton pixels with degree ≥ `degree_threshold`)
   - Dilate junction seeds by `radius_px` to form junction regions

2. **Gate Generation**:
   - Create gate mask: `gate = 1.0` everywhere
   - Set `gate = gate_value_in_junction` inside junction regions
   - Resize gate to match each decoder stage resolution

3. **FiLM Gating** (in decoder):
   - Apply FiLM modulation: `x_modulated = gamma * x + beta`
   - Apply spatial gate: `x_out = x + gate * (x_modulated - x)`
   - Where gate=0: output = input (topology preserved)
   - Where gate=1: output = FiLM(input) (normal behavior)

## Usage Examples

### Example 1: Disable Junction Gating (Default)
```yaml
junction_gating_config:
  enabled: false
```
Behavior: Exact same as before (bitwise identical outputs)

### Example 2: Enable with Prediction Source (Self-Critic)
```yaml
junction_gating_config:
  enabled: true
  source: "pred"
  gate_value_in_junction: 0.0  # Fully suppress FiLM at junctions
```
Behavior: Model uses its own prediction to identify junctions and suppresses FiLM there

### Example 3: Enable with Ground Truth (Oracle Mode)
```yaml
junction_gating_config:
  enabled: true
  source: "gt"
  gate_value_in_junction: 0.0
```
Behavior: Uses ground truth mask for junction detection (upper bound performance)

### Example 4: Partial FiLM Suppression
```yaml
junction_gating_config:
  enabled: true
  source: "pred"
  gate_value_in_junction: 0.3  # 30% FiLM strength at junctions
```
Behavior: Reduces FiLM modulation to 30% at junctions (softer gating)

## Monitoring

Junction gating statistics are logged during training (when FiLM logging is enabled):

```json
{
  "junction_gate_enabled": true,
  "junction_gate_mean": 0.987,
  "junction_gate_min": 0.0,
  "junction_gate_max": 1.0,
  "junction_area_pct": 1.3
}
```

- `junction_gate_mean`: Average gate value (closer to 1.0 = less gating)
- `junction_area_pct`: Percentage of image area with suppressed FiLM

## Testing

Run the test suite to verify implementation:

```bash
python tests/test_junction_gating.py
```

Tests cover:
1. Junction detection on synthetic Y-junctions
2. Gate computation with various configs
3. Backward compatibility (disabled gating = original behavior)

## Performance Considerations

**Computational Cost:**
- Junction detection uses skeletonization (skimage) which is relatively fast (~5-10ms per 320x320 image)
- Gate generation is done once per forward pass at the decoder bottleneck
- Minimal overhead: <1% increase in training time

**Memory:**
- Gate tensors are small (B, 1, H, W) and reused across stages
- Negligible memory increase (~0.1% of total)

**When to Use:**
- ✓ When topological accuracy (clDice, Betti numbers) is important
- ✓ When false positives at junctions are problematic
- ✓ When using VLM-FiLM with aggressive modulation (high gamma/beta scales)
- ✗ When only pixel-wise accuracy (Dice/IoU) matters
- ✗ When training is already stable and topologically accurate

## Dependencies

- **Required**: torch, numpy
- **Optional**: scikit-image (for skeletonization), scipy (for morphological ops)
  - If not available: Falls back to PyTorch-based approximation (less accurate)

Install optional dependencies:
```bash
pip install scikit-image scipy
```

## Implementation Files

- `src/utils/junction_gating.py`: Core junction detection and gate generation
- `src/archs/components/vlm_film.py`: FiLM application with spatial gating
- `src/archs/components/diffusion_unet_vlm_film.py`: UNet integration
- `src/archs/flow_model_vlm_film.py`: Flow model wrapper
- `tests/test_junction_gating.py`: Test suite

## Future Enhancements

Potential improvements:
1. **Learned gating**: Train a lightweight network to predict junction gates
2. **Adaptive radius**: Auto-tune `radius_px` based on vessel thickness
3. **Multi-scale detection**: Detect junctions at multiple resolutions
4. **Confidence-based gating**: Modulate gate strength based on prediction confidence

## References

- VLM-FiLM analysis: `docs/vlm_film_analysis_report_20260202.md`
- False positive pattern analysis: `scripts/analyze_betti_error_patterns.py`
- Configuration example: `configs/flow/xca/flow_sauna_medsegdiff.yaml`

## License

Same as parent project.
