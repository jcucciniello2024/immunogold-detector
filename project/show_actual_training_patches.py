"""Show actual training patches that the model sees (512x512 with gold particles)."""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

from dataset_points import _to_chw_01, PointPatchDataset
from prepare_labels import discover_image_records
from augmentations import (
    ElasticDeform, GaussianBlur, GammaCorrection,
    BrightnessContrast, GaussianNoise, SaltPepperNoise,
    Cutout, apply_augmentation
)

def normalize_for_display(img):
    """Normalize for display."""
    if img.ndim == 3:
        img = np.mean(img, axis=0)
    img = np.asarray(img, dtype=np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img

def main():
    print("Loading real training dataset...")

    # Discover records
    records = discover_image_records("data/Max Planck Data/Gold Particle Labelling/analyzed synapses")
    print(f"Found {len(records)} EM images")

    # Create dataset to get actual training patches
    train_ds = PointPatchDataset(
        records,
        patch_size=(512, 512),
        samples_per_epoch=5,
        pos_fraction=1.0,  # Only positive samples with particles
        sigma=2.5,
        target_type="gaussian",
        augment=False,  # Get original first
        seed=42
    )

    print(f"Dataset created. Getting training patches...")

    # Get 5 original patches with particles
    original_patches = []
    for i in range(min(5, len(train_ds))):
        img, hm = train_ds[i]
        original_patches.append(img.numpy())

    if not original_patches:
        print("ERROR: No training patches found!")
        return

    print(f"Got {len(original_patches)} patches. Creating presentation...")

    # Create figure - show one patch with all augmentations
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Use first patch as example
    patch = original_patches[0]
    heatmap = np.zeros((2, 512, 512), dtype=np.float32)
    rng = np.random.default_rng(42)

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(normalize_for_display(patch), cmap='gray')
    ax.set_title('Original Training Patch\n(512×512, with gold particles)',
                 fontsize=11, fontweight='bold', color='darkgreen')
    ax.axis('off')

    # Elastic Deformation
    ax = fig.add_subplot(gs[0, 1])
    elastic = ElasticDeform(alpha=30, sigma=5)
    img_aug, _ = elastic(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('+ Elastic Deform\n(specimen drift)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Gaussian Blur
    ax = fig.add_subplot(gs[0, 2])
    blur = GaussianBlur(sigma_range=(0.5, 2.0))
    img_aug, _ = blur(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('+ Gaussian Blur\n(focus variation)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Gamma Correction
    ax = fig.add_subplot(gs[1, 0])
    gamma = GammaCorrection(gamma_range=(0.75, 1.35))
    img_aug, _ = gamma(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('+ Gamma Correction\n(beam intensity)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Brightness & Contrast
    ax = fig.add_subplot(gs[1, 1])
    bc = BrightnessContrast()
    img_aug, _ = bc(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('+ Brightness/Contrast\n(amplifier gain)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Gaussian Noise
    ax = fig.add_subplot(gs[1, 2])
    noise = GaussianNoise(sigma_range=(0.01, 0.04))
    img_aug, _ = noise(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('+ Gaussian Noise\n(detector noise)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Salt & Pepper
    ax = fig.add_subplot(gs[2, 0])
    sp = SaltPepperNoise(fraction=0.001)
    img_aug, _ = sp(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('+ Salt & Pepper\n(cosmic rays)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Cutout
    ax = fig.add_subplot(gs[2, 1])
    cutout = Cutout(size_frac=1.0/20.0, max_count=1)
    img_aug, _ = cutout(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('+ Cutout\n(dust particles)', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Full pipeline
    ax = fig.add_subplot(gs[2, 2])
    img_aug, _ = apply_augmentation(patch.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Full Pipeline\n(all augmentations)', fontsize=10, fontweight='bold', color='red')
    ax.axis('off')

    # Show additional original patches for comparison
    for i, orig_patch in enumerate(original_patches[1:]):
        if i < 3:
            ax = fig.add_subplot(gs[3, i])
            ax.imshow(normalize_for_display(orig_patch), cmap='gray')
            ax.set_title(f'Other Training Patches', fontsize=10)
            ax.axis('off')

    plt.suptitle('Actual Training Data: 512×512 Patches with Gold Particles\n' +
                 'Model trains on thousands of augmented versions of these patches',
                 fontsize=14, fontweight='bold', y=0.995)

    output_file = 'training_patches_with_augmentations.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Shows actual training patches (512×512) with gold particles")
    print(f"✓ Each patch augmented in 8+ ways during training")

if __name__ == "__main__":
    main()
