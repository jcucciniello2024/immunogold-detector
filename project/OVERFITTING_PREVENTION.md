# Overfitting Prevention & CLAHE Fix

## Problem Summary

1. **CLAHE too aggressive**: clip_limit=2.0 on 64×64 tiles is over-enhancing gold particles, washing them into white blobs. This destroys subtle intensity information needed for detection.

2. **No overfitting detection**: Training was running 100 epochs without checking if model was overfitting.

3. **No early stopping**: Training wouldn't stop even if validation loss plateaued.

---

## Solution 1: CLAHE Aggressiveness Fix

### What Changed

**Old (Job 4594733):**
- `--use_clahe` enabled
- `CLAHEPreprocess(tile_size=64, clip_limit=2.0)`
- Result: Over-enhanced, particles lose definition

**New (Job with no_clahe script):**
- `--use_clahe` DISABLED (USE_CLAHE=false)
- CLAHE preprocessing removed entirely
- Keep all other augmentations (realistic EM perturbations)
- Result: Preserve raw particle intensities + augmentation robustness

### Next Step: Submit New Job

```bash
cd /mnt/beegfs/home/asahai2024/max-planck-project/project
sbatch hpc/train_detector_2d_no_clahe.slurm
# This automatically waits for job 4594733 to complete (dependency=afterok:4594733)
```

---

## Solution 2: Early Stopping & Overfitting Detection

### Added Features

**New arguments in train_detector.py:**
```bash
--early_stop_patience 10         # Stop if val loss doesn't improve for 10 epochs
--early_stop_delta 1e-5          # Minimum improvement to reset patience counter
```

**What the code does now:**

Every epoch, it reports:
```
Epoch 098/100 train=0.006841 val=0.011619 train_pred_mean=0.069361 ...
Epoch 099/100 train=0.006701 val=0.011542 ⚠️  OVERFITTING DETECTED (train loss << val loss)
```

The warning ⚠️ triggers when: `train_loss < 0.5 * val_loss`

If validation loss doesn't improve by at least `--early_stop_delta` for `--early_stop_patience` consecutive epochs:
```
🛑 Early stopping triggered (patience 10 reached)
```

### Configure Early Stopping in SLURM

Edit `train_detector_2d_no_clahe.slurm`:

```bash
# More aggressive: stop sooner
EARLY_STOP_PATIENCE=5          # Stop after 5 epochs with no improvement
EARLY_STOP_DELTA=1e-6          # Stricter: only ~0.000001 improvement counts

# More patient: let training continue longer
EARLY_STOP_PATIENCE=20         # Allow up to 20 epochs of no improvement
EARLY_STOP_DELTA=1e-4          # Looser: improvements of 0.0001+ reset counter
```

---

## Solution 3: Full Overfitting Prevention Strategy

Your pipeline now has **multiple layers** to prevent overfitting:

| Layer | What It Does | Current Setting |
|-------|-------------|-----------------|
| **Data Augmentation** | 8+ realistic EM perturbations | Always on (elastic deform, blur, noise, etc.) |
| **Dropout** | Randomize activations at bottleneck | p=0.1 (subtle) |
| **BatchNorm** | Normalize layer activations | On in every conv layer |
| **Weight Decay** | Penalize large weights | L2 weight_decay=1e-4 |
| **Consistency Loss** | Regularize via augmentation invariance | consistency_weight=0.1 |
| **Overfitting Detection** | Print warning when train loss << val loss | Triggers when ratio < 0.5 |
| **Early Stopping** | Stop training if val loss plateaus | patience=10 (new) |

---

## How to Check Current Training Status

### SSH into HPC cluster:

```bash
ssh asahai2024@login.cluster.de    # Or your HPC login

# Check job status
squeue -u $USER
sacct -j 4594733 -o jobid,state,elapsed,ntasks,ncpus

# View training output (last 50 lines)
tail -50 /mnt/beegfs/home/asahai2024/max-planck-project/project/logs/gold_detector2d_*.out

# Check if eval job started (4594734)
squeue -u $USER | grep 4594734
```

### What to look for in logs:

```
✓ Normal training progression:
Epoch 001/100 train=0.5000 val=0.4800
Epoch 002/100 train=0.3500 val=0.3400
Epoch 003/100 train=0.2100 val=0.2200  (val slightly higher = healthy)

✗ Overfitting pattern:
Epoch 050/100 train=0.0010 val=0.1500  ⚠️  OVERFITTING (huge gap!)
Epoch 051/100 train=0.0008 val=0.1600  (train keeps dropping, val keeps rising)

✓ Early stopping triggers:
Epoch 042/100 train=0.0050 val=0.0100
Epoch 043/100 train=0.0048 val=0.0101  (no improvement)
Epoch 044/100 train=0.0047 val=0.0102  (no improvement)
...
Epoch 052/100 train=0.0045 val=0.0103  (no improvement)
🛑 Early stopping triggered (patience 10 reached)
```

---

## Expected Metrics

After job 4594734 (evaluation) completes, you'll get:

```
EVALUATION METRICS
==================

6nm particles:
  Precision: 0.XX    (of detected particles, how many are real)
  Recall:    0.XX    (of actual particles, how many did we find)
  F1 Score:  0.XX    (harmonic mean of precision/recall)
  Mean Localization Error: X.XX px

12nm particles:
  Precision: 0.XX
  Recall:    0.XX
  F1 Score:  0.XX
  Mean Localization Error: X.XX px

OVERALL (both types):
  F1 Score:  0.XX   ← This is your main metric
```

**Target goals:**
- No CLAHE: F1 should improve vs current 0.0006
- Early stopping should prevent overfitting
- Val loss should stabilize, not diverge from train loss

---

## Debugging if Metrics Are Still Low

1. **Check train/val split**: Are we using consistent random seed?
2. **Verify data loading**: Run `python show_actual_training_patches.py` to confirm patches are real
3. **Check loss curves**: If train loss doesn't decrease, learning rate might be too low
4. **Inspect model outputs**: Heatmaps should have peaks at particle locations
5. **Consider data quality**: Labels might have inconsistencies

---

## Action Items

- [ ] SSH to HPC and check logs from job 4594733
- [ ] Wait for job 4594733 to complete
- [ ] Submit `train_detector_2d_no_clahe.slurm` (automatically waits for 4594733)
- [ ] Monitor job 4594734 (evaluation) for final metrics
- [ ] Compare F1 scores: with CLAHE vs without CLAHE
