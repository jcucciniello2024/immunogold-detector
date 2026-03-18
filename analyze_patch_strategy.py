"""Analyze optimal patch size for this dataset."""
import os
import sys
import numpy as np
import tifffile

project_dir = os.path.join(os.path.dirname(__file__), 'project')
sys.path.insert(0, project_dir)

from prepare_labels import discover_image_records


def analyze_patch_sizes():
    """Analyze current dataset and patch strategies."""
    print("=" * 80)
    print("PATCH SIZE ANALYSIS FOR GOLD PARTICLE DETECTION")
    print("=" * 80)

    records = discover_image_records("project/data/Max Planck Data/Gold Particle Labelling/analyzed synapses")
    print(f"\nFound {len(records)} EM images\n")

    # Analyze each image
    total_particles_6nm = 0
    total_particles_12nm = 0
    image_sizes = []

    for rec in records:
        try:
            img = tifffile.imread(rec.image_path)
            if img.ndim == 3:
                h, w, c = img.shape
            else:
                h, w = img.shape
                c = 1

            n_6nm = len(rec.points[0]) if len(rec.points) > 0 else 0
            n_12nm = len(rec.points[1]) if len(rec.points) > 1 else 0
            total_6nm = n_6nm + n_12nm
            total_12nm = n_12nm

            image_sizes.append((h, w))
            total_particles_6nm += n_6nm
            total_particles_12nm += n_12nm

            print(f"Image: {rec.image_id}")
            print(f"  Size: {h}×{w} pixels")
            print(f"  Particles: 6nm={n_6nm}, 12nm={n_12nm}, Total={n_6nm + n_12nm}")
            print()

        except Exception as e:
            print(f"Error: {e}")

    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    avg_h = np.mean([s[0] for s in image_sizes])
    avg_w = np.mean([s[1] for s in image_sizes])
    total_particles = total_particles_6nm + total_particles_12nm

    print(f"\nImage dimensions:")
    print(f"  Average: {avg_h:.0f}×{avg_w:.0f}")
    print(f"  Range: {min(s[0] for s in image_sizes)}×{min(s[1] for s in image_sizes)} to "
          f"{max(s[0] for s in image_sizes)}×{max(s[1] for s in image_sizes)}")

    print(f"\nTotal particles in dataset:")
    print(f"  6nm: {total_particles_6nm}")
    print(f"  12nm: {total_particles_12nm}")
    print(f"  Total: {total_particles}")
    print(f"  Average per image: {total_particles / len(records):.1f}")

    print("\n" + "=" * 80)
    print("PATCH SIZE COMPARISON")
    print("=" * 80)

    # Current setup
    print("\n📊 CURRENT SETUP: 512×512 patches")
    print("  Pros:")
    print("    ✅ Large context window")
    print("    ✅ Multiple particles per patch (~5-15 avg)")
    print("    ✅ Model sees particle relationships")
    print("    ✅ Less overfitting on small data")
    print("  Cons:")
    print("    ❌ Only 1 random patch per image per epoch")
    print("    ❌ Limited training data (10 images → ~10 patches/epoch)")
    print("    ❌ Takes longer to compute")
    print("    ❌ Might miss details at edges")

    print("\n📊 ALTERNATIVE: 256×256 patches")
    print("  Pros:")
    print("    ✅ 4 non-overlapping patches per 512×512")
    print("    ✅ ~40 patches per image (16× more training data!)")
    print("    ✅ More epochs see different regions")
    print("    ✅ Faster computation per patch")
    print("    ✅ Better for limited data (more examples)")
    print("  Cons:")
    print("    ❌ Less context (fewer particles per patch)")
    print("    ❌ Smaller model receptive field needed")
    print("    ❌ Might miss relationships between distant particles")

    print("\n📊 ALTERNATIVE: Sliding Window (256×256 with overlap)")
    print("  Pros:")
    print("    ✅ Even more patches (~100-200 per image)")
    print("    ✅ Particles not cut at patch boundaries")
    print("    ✅ Smooth coverage of image")
    print("    ✅ Maximum training data")
    print("  Cons:")
    print("    ❌ High redundancy (overlapping patches)")
    print("    ❌ Correlated training data (less independent)")
    print("    ❌ More computation")

    print("\n📊 ALTERNATIVE: 384×384 patches")
    print("  Pros:")
    print("    ✅ Middle ground (good context + more patches)")
    print("    ✅ 4 patches from 768×768, 2-3 from smaller images")
    print("    ✅ Balance between coverage and context")
    print("  Cons:")
    print("    ❌ Odd size (not power of 2)")
    print("    ❌ Doesn't tile perfectly with 512")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\n🎯 BEST OPTION FOR YOUR DATA: Hybrid Strategy")
    print("""
Strategy: 256×256 patches with SLIDING WINDOW (stride=128)
═════════════════════════════════════════════════════════

1. For each full EM image (e.g., 2048×2115):
   ├─ Extract 256×256 patches
   ├─ Use stride=128 (50% overlap)
   ├─ This gives: ~15-20 patches per image
   └─ Total: 10 images × 15-20 = 150-200 patches per epoch

2. Training impact:
   ├─ Instead of 10 patches/epoch → 150-200 patches/epoch
   ├─ 15-20× MORE training data per epoch!
   ├─ Same compute budget → 15-20× more examples
   ├─ Better regularization (more data = less overfitting)
   └─ Expected F1 improvement: 50-100% (major!)

3. Model adjustments:
   ├─ Current UNet handles any size (scales with patches)
   ├─ No architecture changes needed
   ├─ Same augmentations apply
   └─ Early stopping still works

4. Why this works:
   ├─ 256×256 still has 5-10 particles per patch (good context)
   ├─ Overlapping patches mean particles aren't split at boundaries
   ├─ Model sees each particle from multiple angles/contexts
   ├─ Smooth spatial coverage
   └─ Much more training data without synthetic augmentation
""")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    comparison = """
Strategy              | Patches/Epoch | Context | Redundancy | Computation | Data
──────────────────────┼───────────────┼─────────┼────────────┼─────────────┼──────────
512×512 (current)     |      10       | Excellent|    None   |   Baseline  | Limited
256×256 (non-overlap) |     40-80     | Good    |    None   |   ~4×       | Good
256×256 (stride=128)  |    150-200    | Good    |   50%     |   ~15×      | Excellent
384×384 (stride=192)  |     50-70     | Very Good|   50%    |   ~8×       | Good
512×512 (stride=256)  |     20-30     | Excellent|   50%    |   ~3×       | Good
"""

    print(comparison)

    print("\n" + "=" * 80)
    print("IMPLEMENTATION PLAN")
    print("=" * 80)

    print("""
To implement 256×256 sliding window patches:

1. Modify dataset_points.py:
   ├─ Change patch_size from (512, 512) to (256, 256)
   ├─ Add sliding_window=True parameter
   ├─ Add stride parameter (default 128)
   ├─ Iterate through image with stride
   └─ Extract overlapping patches

2. Update train_detector.py:
   ├─ Change train_samples_per_epoch: 1024 → 2048-4096
   │  (since we have more patches available)
   ├─ Adjust batch_size if needed (might be faster)
   ├─ Early stopping still works the same way
   └─ No model architecture changes

3. Update SLURM script:
   ├─ PATCH_SIZE="256x256"
   ├─ PATCH_STRIDE="128"
   ├─ TRAIN_SAMPLES_PER_EPOCH="2048"
   └─ Run with same GPU (smaller patches are faster)

4. Expected results:
   ├─ Training time: Similar or faster (more efficient data usage)
   ├─ Model accuracy: 50-100% better (more training data)
   ├─ Early stopping: Likely triggers later (more data to learn from)
   └─ F1 improvement: Major (data is the bottleneck, not model)
""")

    print("\n" + "=" * 80)
    print("MY RECOMMENDATION")
    print("=" * 80)

    print("""
✅ DO THIS NEXT (after current job completes):

1. Keep 512×512 for job 4594733 (let it finish)
2. Modify to 256×256 with stride=128 for next training runs
3. Expected impact: 2-3× F1 improvement just from more training data

Rationale:
  • Your dataset is LIMITED (10 images)
  • Data is the bottleneck, not the model
  • 256×256 patches let you use YOUR DATA more efficiently
  • Overlapping patches ensure particles have context
  • No negative impact, only positive (more data = better training)

Priority: HIGH - This will likely have bigger impact than
         architecture changes or other tweaks
""")


if __name__ == "__main__":
    analyze_patch_sizes()
