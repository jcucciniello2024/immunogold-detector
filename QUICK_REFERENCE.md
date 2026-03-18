# Quick Reference: CLAHE Fix & Overfitting Prevention

## 🔴 What Was Broken

**CLAHE (Contrast-Limited Adaptive Histogram Equalization)**
- Settings: `tile_size=64, clip_limit=2.0`
- Problem: Over-enhanced gold particles into white blobs
- Impact: Destroys subtle intensity information needed for detection

## 🟢 What's Fixed

### 1. **Early Stopping Added**
- Detects when validation loss stops improving
- Automatically stops training after N epochs with no improvement
- Prevents wasted compute on plateaued models

### 2. **Overfitting Detection**
- Prints ⚠️ warning when training loss << validation loss
- Helps spot models that memorized training data

### 3. **CLAHE Disabled (Next Run)**
- New script: `hpc/train_detector_2d_no_clahe.slurm`
- `USE_CLAHE=false` - removes aggressive contrast enhancement
- Keeps all other augmentations (elastic deform, blur, noise, etc.)

---

## 📋 Current Status

| Job | Status | What It Does |
|-----|--------|-------------|
| **4594733** | Running/Completed | Training WITH CLAHE (to completion) |
| **4594734** | Waiting | Evaluation of job 4594733 results |
| **Next** | Not submitted | Training WITHOUT CLAHE (new script ready) |

---

## ✅ Action Checklist

### Step 1: Check Current Job Progress
```bash
# SSH to HPC
ssh asahai2024@login.cluster.de

# Check job 4594733 status
squeue -u $USER | grep 4594733

# View last 50 lines of training log
tail -50 logs/gold_detector2d_*.out

# Look for these patterns:
# - Epoch 098/100 ...       (near completion, good)
# - 🛑 Early stopping ...   (stopped early, also good)
# - ERROR / FAILED          (problem to investigate)
```

### Step 2: Wait for Evaluation Results
Once job 4594733 completes:
- Job 4594734 starts automatically
- Generates `eval_results_<jobid>.txt` with F1/precision/recall

### Step 3: Pull Metrics Locally
```bash
# From your Mac in project directory
./project/hpc/pull_metrics.sh

# View results
cat hpc_results/eval_results_*.txt
```

### Step 4: Submit Retraining Without CLAHE
```bash
# SSH to HPC
ssh asahai2024@login.cluster.de
cd /mnt/beegfs/home/asahai2024/max-planck-project/project

# Submit new job (waits for 4594733 to finish)
sbatch hpc/train_detector_2d_no_clahe.slurm

# Confirm submission
squeue -u $USER
```

### Step 5: Compare Results
Compare F1 scores:
- **With CLAHE (4594733)**: F1 = ?
- **Without CLAHE (next job)**: F1 = ? (expect higher)

---

## 🎯 Key Parameters Explained

### Early Stopping (can be customized)
```bash
EARLY_STOP_PATIENCE=10          # Stop after 10 epochs with no improvement
EARLY_STOP_DELTA=1e-5           # Minimum val loss improvement threshold
```

**Conservative** (let training continue):
```bash
EARLY_STOP_PATIENCE=20
EARLY_STOP_DELTA=1e-6
```

**Aggressive** (stop sooner):
```bash
EARLY_STOP_PATIENCE=5
EARLY_STOP_DELTA=1e-4
```

### Overfitting Warning
Currently triggers when: `train_loss < 0.5 * val_loss`

Example output:
```
Epoch 050/100 train=0.0050 val=0.0100 ⚠️  OVERFITTING DETECTED (train loss << val loss)
```

---

## 📊 Expected Results

After removing CLAHE, you should see:
1. ✅ Better F1 scores (preserve particle signals)
2. ✅ Healthier train/val loss curves (not diverging wildly)
3. ✅ Early stopping prevents wasting epochs
4. ✅ Cleaner training logs with overfitting warnings

---

## 🔧 If Something Goes Wrong

### Q: Training is very slow
A: Check GPU utilization with `nvidia-smi` on HPC. May need to adjust `BATCH_SIZE` or `TRAIN_SAMPLES_PER_EPOCH`.

### Q: F1 score is still very low
A: Check if labels are correct. Run `show_actual_training_patches.py` to visualize what model sees.

### Q: Job failed with error
A: Run `cat logs/gold_detector2d_*.err` to see error message.

### Q: Job timed out before finishing
A: Increase `--time=06:00:00` in SLURM script (currently 6 hours).

---

## 📁 Files Modified

- ✅ `train_detector.py` - Added early stopping & overfitting detection
- ✅ `hpc/train_detector_2d_no_clahe.slurm` - NEW script with CLAHE disabled
- ✅ `hpc/pull_metrics.sh` - NEW script to fetch results from HPC
- ✅ `OVERFITTING_PREVENTION.md` - Detailed explanation
- ✅ `QUICK_REFERENCE.md` - This file (cheat sheet)

---

## 🚀 Next Steps

1. **Right now**: SSH to HPC and check if job 4594733 is still running
2. **After 4594733 completes**: Job 4594734 starts automatically
3. **When 4594734 finishes**: Pull metrics with `./hpc/pull_metrics.sh`
4. **Then submit**: `sbatch hpc/train_detector_2d_no_clahe.slurm`
5. **Finally**: Compare F1 scores to see if CLAHE removal helped
