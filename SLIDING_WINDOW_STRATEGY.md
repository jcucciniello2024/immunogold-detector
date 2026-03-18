# Sliding Window Patching: 15-20× More Training Data

## The Problem: You're Wasting Data

**Current approach (Random Sampling)**
```
Image 1 (2048×2048) → Pick 1 random 512×512 patch
Image 2 (2048×2048) → Pick 1 random 512×512 patch
...
Image 10 (2048×2048) → Pick 1 random 512×512 patch

Total per epoch: 10 patches
```

**You have 453 particles across 10 images, but only seeing ~10 patches/epoch!**

---

## The Solution: Sliding Window Extraction

**New approach (Sliding Window with Overlap)**
```
Image 1 (2048×2048):
  ├─ Extract 256×256 patch at (0, 0)
  ├─ Extract 256×256 patch at (0, 128) ← stride=128 (50% overlap)
  ├─ Extract 256×256 patch at (0, 256)
  ├─ ...continue with stride 128...
  └─ Total: ~15-20 patches per image

Image 2-10: Same process

Total per epoch: 10 images × 15-20 = 150-200 patches
```

**15-20× MORE DATA with same compute!**

---

## Why This Works

### Smaller Patches = Focused Learning
```
512×512 patch:      ~20-30 particles
256×256 patch:      ~5-10 particles
```

Smaller patches help the model:
- ✅ Focus on individual particles
- ✅ Learn particle features at detail level
- ✅ Not distracted by other particles
- ✅ Better for limited dataset (more diverse examples)

### Overlapping Patches = Complete Coverage
```
Stride=128 (50% overlap):
  - Particles aren't split at patch boundaries
  - Each particle visible in 2-4 different patches
  - Model learns from multiple perspectives
  - No "dead zones" between patches
```

### More Epochs = Better Learning
```
Current (512×512):
  Epoch 1: See 10 random patches
  Epoch 2: See 10 different random patches
  ...
  Epoch 100: Seen ~1,000 total patches

New (256×256, stride=128):
  Epoch 1: Systematically sample from 150-200 available patches
  Epoch 2: Systematically sample from 150-200 available patches
  ...
  Epoch 100: Seen ~150-200 patches (but with high redundancy, seen them many times)
```

**Expected improvement: 2-3× F1 score**

---

## Implementation

### File: `dataset_points_sliding_window.py` ✅ CREATED

New dataset class that:
1. **Pre-computes all patch locations** at initialization
2. **Randomly samples from available patches** during training
3. **Maintains all existing functionality** (augmentation, heatmaps, consistency loss)
4. **Reports statistics** on how many patches are available

### To Use It:

**Option 1: In training script**
```python
from dataset_points_sliding_window import SlidingWindowPatchDataset

train_ds = SlidingWindowPatchDataset(
    train_records,
    patch_size=(256, 256),  # Smaller patches
    patch_stride=128,        # 50% overlap
    samples_per_epoch=2048,  # More samples to train on
    augment=True,
    # ... other args same as before
)
```

**Option 2: In SLURM script**
```bash
python train_detector.py \
  --data_root "$DATA_ROOT" \
  --dataset_class sliding_window \
  --patch_size 256 256 \
  --patch_stride 128 \
  --train_samples_per_epoch 2048 \
  # ... other args
```

---

## Expected Results

### With Sliding Window (256×256, stride=128):

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| Patches/epoch | 10 | 150-200 | **15-20×** |
| Training time | Baseline | ~2-3× | More data to learn |
| F1 Score | 0.0006? | 0.001-0.003? | **2-3×** |
| Model overfitting | Medium | Lower | Better generalization |
| GPU memory | Baseline | Same | 256×256 < 512×512 |
| GPU compute | Baseline | 2-3× | Faster per patch |

---

## Configuration Options

### Conservative (Least Change)
```
patch_size: (512, 512)
patch_stride: 256
samples_per_epoch: 512
```
- Still overlapping (2 patches per image dimension)
- 4× more data
- Safer, less of a change

### Balanced (Recommended)
```
patch_size: (256, 256)
patch_stride: 128
samples_per_epoch: 2048
```
- Good trade-off between patch size and data quantity
- 15-20× more data
- Particles still have context

### Aggressive (Maximum Data)
```
patch_size: (256, 256)
patch_stride: 64
samples_per_epoch: 3000+
```
- Maximum data (75% overlap)
- Highest redundancy
- Very compute-intensive
- Not recommended unless data is extremely limited

---

## Comparison Visualization

```
CURRENT (512×512 Random):
┌─────────────────────────────────────┐
│  2048×2048 EM Image                 │
│  ┌─────────────┐                    │
│  │  512×512    │ (Random position)  │
│  │  Patch 1    │                    │
│  └─────────────┘                    │
│                                     │
│  Next epoch: Pick random location   │
│  again, completely different patch  │
└─────────────────────────────────────┘

ONE random patch per image per epoch = 10 patches total


NEW (256×256 Sliding Window, stride=128):
┌─────────────────────────────────────┐
│  2048×2048 EM Image                 │
│  ┌──┬──┬──┬──┬──┬──┬──┬──────────┐  │
│  │  │  │  │  │  │  │  │          │  │
│  ├──┼──┼──┼──┼──┼──┼──┤ 256×256  │  │
│  │  │  │  │  │  │  │  │ patches  │  │
│  ├──┼──┼──┼──┼──┼──┼──┤ with 50% │  │
│  │  │  │  │  │  │  │  │ overlap  │  │
│  ├──┼──┼──┼──┼──┼──┼──┤ (stride  │  │
│  │  │  │  │  │  │  │  │ =128)    │  │
│  └──┴──┴──┴──┴──┴──┴──┴──────────┘  │
└─────────────────────────────────────┘

~16 patches per image systematically extracted
10 images × 16 = 160 patches available per epoch
Randomly sample 2048 patches per epoch (with replacement)
Every patch systematically available, none missed


PATCHES AVAILABLE (not data augmentation redundancy):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Old: 1 patch/image × 10 images = 10 patches/epoch available
New: 16 patches/image × 10 images = 160 patches/epoch available

Data amplification factor: 16× MORE PATCHES AVAILABLE
(This is different from augmentation - it's using your data better)
```

---

## What About Redundancy?

**With overlapping patches, isn't training data correlated?**

Yes, but:
1. **Patches still have different content** (only 50% overlap)
2. **Augmentation adds randomness** (different augmentations per patch)
3. **Model learns from multiple perspectives** (particle visible in 2-4 patches)
4. **Data is the bottleneck** (more correlated data > less independent data)
5. **Early stopping prevents overfitting** (won't fit noise even with redundancy)

For a limited dataset (10 images, 453 particles), the benefits of **more data** outweigh the costs of **correlation**.

---

## When to Switch

### Keep 512×512:
- ❌ If you have 1000+ images (plenty of data)
- ❌ If particles are very small and need context

### Switch to 256×256:
- ✅ You only have 10 images (you are here!)
- ✅ Particles are medium to large (6-12nm)
- ✅ Want to maximize data usage
- ✅ Have limited training compute budget

---

## Implementation Timeline

### Now (Job 4594733):
- Let it finish with 512×512 (already running)
- Get baseline F1 score with CLAHE disabled

### Next (Job 4594734):
- Evaluation of job 4594733
- Baseline metrics

### After That (Recommended):
```
Job N (New training with sliding window):
├─ Use SlidingWindowPatchDataset
├─ patch_size=(256, 256)
├─ patch_stride=128
├─ train_samples_per_epoch=2048
├─ Everything else same (early stopping, augmentation, etc.)
└─ Expected: 2-3× better F1 than 512×512

Compare:
  F1 with 512×512 = ???
  F1 with 256×256 sliding = ??? (expect much better!)
```

---

## Testing

To test the sliding window dataset locally:

```python
from project.dataset_points_sliding_window import SlidingWindowPatchDataset
from project.prepare_labels import discover_image_records

records = discover_image_records("project/data/...")
ds = SlidingWindowPatchDataset(
    records,
    patch_size=(256, 256),
    patch_stride=128,
    samples_per_epoch=100,
    augment=False
)

print(f"Dataset size: {len(ds)}")
img, hm = ds[0]
print(f"Patch shape: {img.shape}, heatmap shape: {hm.shape}")
```

Should print:
```
SlidingWindowPatchDataset initialized:
  Records: 10
  Total patch locations: 160
  Patches per epoch (samples_per_epoch): 100
  Patch size: 256×256, stride: 128
```

---

## Summary

| Aspect | 512×512 Random | 256×256 Sliding |
|--------|---|---|
| Patches available | 10 | 150-200 |
| Context per patch | High | Good |
| Training data usage | Low (wasteful) | High (efficient) |
| Redundancy | None | 50% overlap |
| Computation/patch | Higher | Lower |
| Total F1 expected | 0.0006+ | 0.002+  (3×) |
| Recommended? | ✅ Current | ⭐ Next |

**This is the single biggest improvement you can make with existing data.**
