# Blood Vessel Segmentation - Research Codebase

A production-ready research codebase for blood vessel segmentation using supervised and diffusion-based deep learning models.

##  Quick Start

### Training
```bash
# Train all 4 models on OCTA500 3M dataset
./scripts/train_octa500_3m.sh

# Monitor with TensorBoard
tensorboard --logdir experiments/ --port 6006 --bind_all
```

### Evaluation
```bash
# Evaluate all trained models
./scripts/eval_octa500_3m.sh

# View results
cat results/octa500_3m/evaluation_octa500_3m.csv
```

## 📋 Features

### Models (4)
- **csnet** - CS-Net (Channel & Spatial attention) - 8.4M params
- **dscnet** - DSCNet (Dual-stage cascaded) - 5.8M params  
- **medsegdiff** - MedSegDiff (Medical segmentation diffusion) - 16.2M params
- **berdiff** - BerDiff (Bernoulli diffusion) - 9.3M params

### Datasets (3)
- **octa500_3m** - OCTA-500 3×3mm (200/50/50 splits)
- **octa500_6m** - OCTA-500 6×6mm (200/50/50 splits)
- **rossa** - ROSSA dataset (35/9/9 splits)

### System Features
- ✅ **Unified Interface** - Same commands for supervised and diffusion models
- ✅ **Auto Experiment Tracking** - Git hash, configs, metrics automatically logged
- ✅ **Multi-GPU Support** - Automatic GPU allocation for parallel training
- ✅ **TensorBoard Integration** - Real-time monitoring with image logging
- ✅ **Best Checkpoint Management** - Automatic best model selection
- ✅ **Dataset-Centric Design** - One command trains/evaluates all models

## 📁 Structure

```
soft-seg/
├── src/
│   ├── registry/          # Model & dataset metadata
│   ├── experiment/        # Experiment tracking
│   └── runner/            # Training & evaluation
├── scripts/               # Executable scripts
│   ├── train.py          # Training CLI
│   ├── evaluate.py       # Evaluation CLI
│   ├── train_*.sh        # Dataset-specific training
│   └── eval_*.sh         # Dataset-specific evaluation
├── experiments/           # All experiment results
└── results/              # Evaluation results
```

## 📖 Documentation

- **[Training Guide](TRAINING_GUIDE.md)** - Complete usage guide
- **[Refactoring Guide](REFACTORING_GUIDE.md)** - System design and architecture

## 💻 Requirements

- Python 3.12+
- PyTorch 2.0+
- Lightning 2.5+
- CUDA-capable GPU (7x RTX A6000 recommended)

## 🔧 Installation

```bash
# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 Training

### Dataset-wise (Recommended)
```bash
# Train all models on each dataset
./scripts/train_octa500_3m.sh   # 4 models on GPU 0-3
./scripts/train_octa500_6m.sh   # 4 models on GPU 0-3
./scripts/train_rossa.sh        # 4 models on GPU 0-3
```

### Single Model
```bash
# Train specific model
uv run python scripts/train.py --model csnet --data octa500_3m --gpu 0

# Adjust batch size
uv run python scripts/train.py --model medsegdiff --data octa500_6m --batch-size 8
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir experiments/ --port 6006 --host 0.0.0.0

# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f logs/train_*.log

# Stop all training
pkill -f train.py
```

## 🧪 Evaluation

### Dataset-wise (Recommended)
```bash
# Evaluate all models on each dataset
./scripts/eval_octa500_3m.sh
./scripts/eval_octa500_6m.sh
./scripts/eval_rossa.sh
```

### Custom Evaluation
```bash
# Specific models only
uv run python scripts/evaluate.py --data octa500_3m --models csnet,dscnet

# Save predictions
uv run python scripts/evaluate.py --data octa500_6m --save-predictions

# Use specific GPU
uv run python scripts/evaluate.py --data rossa --gpu 1
```

## 🎯 Typical Workflow

```bash
# 1. Train all models
./scripts/train_octa500_3m.sh

# 2. Monitor progress
tensorboard --logdir experiments/ --port 6006 --bind_all
# Open: http://localhost:6006

# 3. After training completes
./scripts/eval_octa500_3m.sh

# 4. Check results
cat results/octa500_3m/evaluation_octa500_3m.csv

# 5. Repeat for other datasets
./scripts/train_octa500_6m.sh && ./scripts/eval_octa500_6m.sh
./scripts/train_rossa.sh && ./scripts/eval_rossa.sh
```

## 📈 Results

Evaluation results are saved as CSV files:
```
results/
├── octa500_3m/
│   └── evaluation_octa500_3m.csv
├── octa500_6m/
│   └── evaluation_octa500_6m.csv
└── rossa/
    └── evaluation_rossa.csv
```

Each CSV contains:
- Model name
- Dice coefficient
- IoU
- Precision, Recall, Specificity
- Experiment ID

## 🗂️ Experiment Management

All experiments are automatically tracked in `experiments/`:

```
experiments/
├── experiments.json          # Experiment database
└── {model}/{dataset}/{run_id}/
    ├── config.yaml          # Training config
    ├── git_info.txt         # Git commit hash
    ├── checkpoints/
    │   └── best.ckpt       # Best checkpoint
    ├── tensorboard/         # TensorBoard logs
    └── summary.json         # Final metrics
```

## 🛠️ Troubleshooting

### GPU Memory Error
```bash
# Reduce batch size
uv run python scripts/train.py --model csnet --data octa500_3m --batch-size 8

# Train models one by one
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --model csnet --data octa500_3m
```

### No Checkpoint Found
```bash
# Check experiment directories
ls -la experiments/{model}/{dataset}/

# Find all checkpoints
find experiments/ -name "best.ckpt"
```

### Training Not Starting
```bash
# Check GPU availability
nvidia-smi

# Check running processes
ps aux | grep train.py

# Check logs
tail -f logs/train_*.log
```

## 📚 Citation

If you use this codebase in your research, please cite:

```bibtex
@software{vessel_segmentation_2025,
  title = {Blood Vessel Segmentation Research Codebase},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yongjoon02/soft-seg}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: The legacy scripts are preserved in `script_legacy/` for reference.
