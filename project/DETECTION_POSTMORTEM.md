# Immunogold Detection Postmortem (Current State)

## Goal

Build a high-rigor immunogold detector for Max Planck EM data with strong F1 performance (eventually publication-grade).

## Current Status

- No run has achieved acceptable F1 yet.
- Best observed binary F1 so far: **0.0059** (0.59%).
- Nothing is remotely close to 90% F1.

## What Was Tried and What Failed

### 1) CenterNet-based detectors

**What we tried**
- CenterNet/CEM500K and enhanced CenterNet variants.
- Loss and AMP stability fixes, shape fixes, dataset discovery fixes, Slurm reliability fixes.

**What happened**
- Initial job instability was fixed (crashes, NaNs, shape mismatch, file discovery issues).
- Models ran, but downstream detection quality remained very poor.

**Why this failed (performance-wise)**
- Even after technical fixes, detector quality/calibration was weak on this dataset.
- High false positives and/or class collapse persisted.

---

### 2) Binary-first U-Net heatmap detector

**What we tried**
- Switched to binary detection first (ignore 6nm vs 12nm initially).
- Gaussian heatmap targets, focal BCE, threshold sweeps.
- Added overlays and heatmap outputs for debugging.

**Measured outcomes**
- `hpc_eval_binary_first_4597392`: best_binary_f1 = **0.0009**
- `hpc_eval_binary_first_4597404`: best_binary_f1 = **0.0009**

**Why this failed**
- Severe precision-recall imbalance from poor peak calibration.
- Either FP flood (too many detections) or over-pruned detections.

---

### 3) Aggressive biologically inspired hard filters

**What we tried**
- Dark center/ring contrast checks.
- Size/anisotropy filters.
- Bottom-bar suppression.
- Stricter decode constraints.

**Measured outcomes**
- Snapshot became over-pruned in some configurations.
- `hpc_eval_binary_snapshot_4597392`: best_binary_f1 = **0.0000**
- Relaxed variant `hpc_eval_binary_snapshot_relaxed_4597392`: best_binary_f1 = **0.0022**

**Why this failed**
- Hard filters removed many false positives but also removed true positives.
- Recall collapsed faster than precision improved.

---

### 4) Baseline simple U-Net pass (minimal fancy filtering)

**What we tried**
- Return to basics: train/infer/eval with fewer constraints.
- Keep overlays/heatmaps to inspect errors.

**Measured outcomes**
- `hpc_eval_baseline_simple_4597439`: best_binary_f1 = **0.0059** (best so far)
- At low thresholds: many detections, very low precision.

**Why this failed**
- Model still confuses tiny texture/background with particles.
- Detection flood remains dominant failure mode.

---

### 5) Gold Digger style cGAN pipeline

**What we tried**
- Trained and evaluated cGAN pipeline (`golddigger_full`).
- Added dedicated GoldDigger overlay/heatmap outputs.

**Measured outcomes**
- `hpc_eval_golddigger_4597447`: best_macro_f1 = **0.0027**

**Why this failed**
- Did not outperform even weak binary baseline.
- Heavy model complexity with unstable benefit under current data regime.

---

### 6) Tiny-particle decode constraints (2-5 px intuition)

**What we tried**
- Local-max decode improvements.
- Tiny support-size gating and strict non-touching constraints at inference.

**Measured outcomes**
- `hpc_eval_tiny_mode_4597439`: best_binary_f1 = **0.0000**
- False positives dropped sharply, but true positives went to near-zero.

**Why this failed**
- Current model confidence peaks are not aligned to true tiny positives.
- Decode tightening alone cannot recover true detection quality.

## Failure Pattern Across All Attempts

1. **Task is tiny-object, low-contrast, high-confound EM detection** (very hard regime).
2. **Model calibration is poor**: heatmap peaks form on wrong micro-texture.
3. **Tradeoff repeatedly breaks**:
   - Relaxed decoding -> FP flood.
   - Strict decoding/filtering -> TP collapse.
4. **Core bottleneck appears to be training signal quality for tiny targets**, not just post-processing.

## What Is Running Now

- `ig_tiny_target` job (ID `4597451`) is currently running.
- This run uses tiny-target training + tiny-particle inference mode to test whether training-side target sharpening improves TP recovery.

## Bottom Line

The project has moved from "pipeline crashes" to "stable but low-quality detection."  
The unresolved problem is now scientific/modeling: learning true tiny immunogold patterns without flooding on EM texture.
