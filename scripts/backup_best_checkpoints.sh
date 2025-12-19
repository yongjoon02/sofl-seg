#!/bin/bash
# Backup best checkpoints of retained models

BACKUP_DIR="archive/old_experiments_backup_20251124"
MODELS=("csnet" "dscnet" "medsegdiff" "berdiff")
DATASETS=("octa500_3m" "octa500_6m" "rossa" "xca")

echo "==================================="
echo "Backing up best checkpoints..."
echo "==================================="

mkdir -p "$BACKUP_DIR"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        CKPT_DIR="lightning_logs/${dataset}/${model}/checkpoints"
        
        if [ -d "$CKPT_DIR" ]; then
            BACKUP_PATH="$BACKUP_DIR/${dataset}/${model}"
            mkdir -p "$BACKUP_PATH"
            
            # Copy best checkpoint if exists
            if [ -f "$CKPT_DIR/best.ckpt" ]; then
                cp "$CKPT_DIR/best.ckpt" "$BACKUP_PATH/"
                echo "✓ Backed up: ${dataset}/${model}/best.ckpt"
            else
                echo "⚠ No checkpoint: ${dataset}/${model}"
            fi
        fi
    done
done

echo ""
echo "==================================="
echo "Backup completed!"
echo "Location: $BACKUP_DIR"
echo "==================================="
du -sh "$BACKUP_DIR"
