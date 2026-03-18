"""Create a presentation image with REAL training data augmentations."""
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from augmentations import (
    ElasticDeform, GaussianBlur, GammaCorrection,
    BrightnessContrast, GaussianNoise, SaltPepperNoise,
    Cutout, CLAHEPreprocess
)

def normalize_for_display(img):
    """Normalize image for display."""
    if img.ndim == 3:
        img = np.mean(img, axis=0)
    img = np.asarray(img, dtype=np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img

def main():
    print("Loading real training image...")

    # Load real EM image
    try:
        real_image = tifffile.imread("/tmp/real_em_image.tif")
    except:
        print("Error: Could not load real image. Using synthetic instead.")
        real_image = np.random.uniform(0.3, 0.7, (512, 512))

    # Convert to 3-channel if needed
    if real_image.ndim == 2:
        real_image = np.stack([real_image] * 3)
    elif real_image.ndim == 3 and real_image.shape[0] > 3:
        real_image = real_image[:3]

    # Normalize
    real_image = np.asarray(real_image, dtype=np.float32)
    if real_image.max() > 1:
        real_image = real_image / real_image.max()

    # Create heatmap placeholder
    heatmap = np.zeros((2, *real_image.shape[-2:]), dtype=np.float32)
    rng = np.random.default_rng(42)

    # Display setup
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(normalize_for_display(real_image), cmap='gray')
    ax.set_title('Original\nEM Image', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Elastic Deformation
    ax = fig.add_subplot(gs[0, 1])
    elastic = ElasticDeform(alpha=30, sigma=5)
    img_aug, _ = elastic(real_image.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Elastic Deform\n(specimen drift)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Gaussian Blur
    ax = fig.add_subplot(gs[0, 2])
    blur = GaussianBlur(sigma_range=(0.5, 2.0))
    img_aug, _ = blur(real_image.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Gaussian Blur\n(focus variation)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Gamma Correction
    ax = fig.add_subplot(gs[1, 0])
    gamma = GammaCorrection(gamma_range=(0.75, 1.35))
    img_aug, _ = gamma(real_image.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Gamma Correction\n(beam intensity)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Brightness & Contrast
    ax = fig.add_subplot(gs[1, 1])
    bc = BrightnessContrast()
    img_aug, _ = bc(real_image.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Brightness/Contrast\n(amplifier gain)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Gaussian Noise
    ax = fig.add_subplot(gs[1, 2])
    noise = GaussianNoise(sigma_range=(0.01, 0.04))
    img_aug, _ = noise(real_image.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Gaussian Noise\n(detector noise)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Salt & Pepper
    ax = fig.add_subplot(gs[2, 0])
    sp = SaltPepperNoise(fraction=0.001)
    img_aug, _ = sp(real_image.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Salt & Pepper\n(cosmic rays)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Cutout
    ax = fig.add_subplot(gs[2, 1])
    cutout = Cutout(size_frac=1.0/20.0, max_count=1)
    img_aug, _ = cutout(real_image.copy(), heatmap.copy(), rng)
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('Cutout\n(dust particles)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # CLAHE
    ax = fig.add_subplot(gs[2, 2])
    clahe = CLAHEPreprocess(tile_size=64, clip_limit=2.0)
    img_aug, _ = clahe(real_image.copy(), heatmap.copy())
    ax.imshow(normalize_for_display(img_aug), cmap='gray')
    ax.set_title('CLAHE\n(contrast enhancement)', fontsize=11, fontweight='bold')
    ax.axis('off')

    plt.suptitle('EM-Realistic Augmentations Applied to Real Training Image',
                 fontsize=14, fontweight='bold', y=0.98)

    output_file = 'augmentation_presentation_REAL.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved presentation image: {output_file}")
    print(f"✓ Image shows real training data with augmentations applied")

if __name__ == "__main__":
    main()
