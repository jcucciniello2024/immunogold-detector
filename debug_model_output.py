"""Debug why F1 scores are so bad - check what model is actually learning."""
import os
import sys
import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_dir = os.path.join(os.path.dirname(__file__), 'project')
sys.path.insert(0, project_dir)

from prepare_labels import discover_image_records
from dataset_points import _to_chw_01, PointPatchDataset
from model_unet_deep import UNetDeepKeypointDetector


def analyze_model_outputs():
    """Analyze what the model is actually outputting."""
    print("=" * 80)
    print("DEBUGGING MODEL OUTPUTS & ACCURACY")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Load data
    records = discover_image_records("project/data/Max Planck Data/Gold Particle Labelling/analyzed synapses")
    print(f"Found {len(records)} EM images")
    print(f"Total particles: {sum(len(r.points[0]) + len(r.points[1]) for r in records)}\n")

    if not records:
        print("ERROR: No records found!")
        return

    # Create dataset
    train_ds = PointPatchDataset(
        records,
        patch_size=(512, 512),
        samples_per_epoch=5,
        pos_fraction=1.0,  # Only positive patches
        augment=False,
        seed=42
    )

    print(f"Dataset size: {len(train_ds)}")

    # Get a few patches
    print("\n" + "=" * 80)
    print("CHECKING TRAINING DATA")
    print("=" * 80)

    for idx in range(min(3, len(train_ds))):
        img, hm = train_ds[idx]
        img_np = img.numpy()
        hm_np = hm.numpy()

        print(f"\nPatch {idx}:")
        print(f"  Image shape: {img_np.shape}, dtype: {img_np.dtype}")
        print(f"  Image range: [{img_np.min():.4f}, {img_np.max():.4f}]")
        print(f"  Heatmap shape: {hm_np.shape}, dtype: {hm_np.dtype}")
        print(f"  Heatmap range: [{hm_np.min():.4f}, {hm_np.max():.4f}]")
        print(f"  6nm heatmap max: {hm_np[0].max():.4f}, pixels > 0.5: {(hm_np[0] > 0.5).sum()}")
        print(f"  12nm heatmap max: {hm_np[1].max():.4f}, pixels > 0.5: {(hm_np[1] > 0.5).sum()}")

    # Try loading a trained model if available
    print("\n" + "=" * 80)
    print("TESTING MODEL OUTPUTS")
    print("=" * 80)

    model = UNetDeepKeypointDetector(in_channels=3, out_channels=2, base_channels=32).to(device)

    # Check model structure
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: UNetDeepKeypointDetector")
    print(f"Total parameters: {total_params:,}")

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {dummy_output.shape}")
    print(f"Output dtype: {dummy_output.dtype}")
    print(f"Output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")

    # Test on real patch
    print("\n" + "=" * 80)
    print("INFERENCE ON REAL PATCH")
    print("=" * 80)

    img, hm_gt = train_ds[0]
    img_batch = img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_batch)
        probs = torch.sigmoid(logits)

    logits_np = logits.cpu().numpy()[0]
    probs_np = probs.cpu().numpy()[0]
    img_np = img.numpy()
    hm_gt_np = hm_gt.numpy()

    print(f"\nGround truth heatmap:")
    print(f"  6nm: max={hm_gt_np[0].max():.4f}, sum={hm_gt_np[0].sum():.2f}, pixels>0.1: {(hm_gt_np[0] > 0.1).sum()}")
    print(f"  12nm: max={hm_gt_np[1].max():.4f}, sum={hm_gt_np[1].sum():.2f}, pixels>0.1: {(hm_gt_np[1] > 0.1).sum()}")

    print(f"\nModel logits:")
    print(f"  6nm: range=[{logits_np[0].min():.4f}, {logits_np[0].max():.4f}]")
    print(f"  12nm: range=[{logits_np[1].min():.4f}, {logits_np[1].max():.4f}]")

    print(f"\nModel probabilities (after sigmoid):")
    print(f"  6nm: range=[{probs_np[0].min():.4f}, {probs_np[0].max():.4f}], mean={probs_np[0].mean():.4f}")
    print(f"  12nm: range=[{probs_np[1].min():.4f}, {probs_np[1].max():.4f}], mean={probs_np[1].mean():.4f}")

    # Visualize
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Normalize for display
    def norm_img(img):
        img = img.astype(np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        return img

    # Input image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(norm_img(np.mean(img_np, axis=0)), cmap='gray')
    ax.set_title('Input Image (512×512)\nMean of 3 channels', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Ground truth 6nm
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(norm_img(np.mean(img_np, axis=0)), cmap='gray', alpha=0.5)
    ax.imshow(hm_gt_np[0], cmap='hot', alpha=0.7)
    ax.set_title('Ground Truth: 6nm\n(overlay on image)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Ground truth 12nm
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(norm_img(np.mean(img_np, axis=0)), cmap='gray', alpha=0.5)
    ax.imshow(hm_gt_np[1], cmap='hot', alpha=0.7)
    ax.set_title('Ground Truth: 12nm\n(overlay on image)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Model output logits 6nm
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(logits_np[0], cmap='RdBu_r')
    ax.set_title('Model Logits: 6nm\n(raw output)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    # Model output logits 12nm
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(logits_np[1], cmap='RdBu_r')
    ax.set_title('Model Logits: 12nm\n(raw output)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Model probabilities 6nm
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(probs_np[0], cmap='hot')
    ax.set_title('Model Probabilities: 6nm\n(after sigmoid)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Model probabilities 12nm
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(probs_np[1], cmap='hot')
    ax.set_title('Model Probabilities: 12nm\n(after sigmoid)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Difference 6nm
    ax = fig.add_subplot(gs[2, 1])
    diff = (probs_np[0] - hm_gt_np[0])
    ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Difference: 6nm\n(pred - gt)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Difference 12nm
    ax = fig.add_subplot(gs[2, 2])
    diff = (probs_np[1] - hm_gt_np[1])
    ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Difference: 12nm\n(pred - gt)', fontsize=11, fontweight='bold')
    ax.axis('off')

    plt.suptitle('Model Output Analysis: Comparing Predictions to Ground Truth\n(Untrained model, so output is basically random)',
                 fontsize=14, fontweight='bold')

    output_file = 'project/model_output_debug.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")

    # Analysis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    diagnosis = """
KEY OBSERVATIONS:
═════════════════

1. UNTRAINED MODEL
   └─ This model has random weights (never trained)
   └─ Output is essentially random noise
   └─ This is EXPECTED for a new model

2. HEATMAP TARGETS
   └─ GT heatmaps have small peaks at particle locations
   └─ Most pixels are ~0 (background)
   └─ Peaks are smooth Gaussians (σ=2.5)
   └─ This is the TARGET the model should learn

3. POTENTIAL ISSUES TO CHECK
   ✓ Are labels correctly aligned with particles?
   ✓ Are heatmap generation using correct coordinates?
   ✓ Is loss function appropriate for small targets?
   ✓ Is learning rate right for this task?
   ✓ Is batch normalization helping or hurting?
   ✓ Are particles visible/clear in the images?

4. WHY F1 IS SO LOW (Likely causes)
   ❌ Model is untrained (start from scratch each time?)
   ❌ Label/image mismatch (wrong coordinates?)
   ❌ Heatmap targets too small/sparse (hard to detect?)
   ❌ Loss function not suitable (class imbalance?)
   ❌ Learning rate too high/low (poor convergence?)
   ❌ Data loading issue (wrong patches/images?)
   ❌ Evaluation metric mismatch (threshold too strict?)

NEXT STEPS:
═════════

1. CHECK LABEL QUALITY
   └─ Visualize labeled particles on images
   └─ Do they align with actual particles?
   └─ Are labels complete/consistent?

2. CHECK HEATMAP GENERATION
   └─ Are heatmaps correctly centered on particles?
   └─ Are peaks at right locations?
   └─ Is sigma reasonable?

3. VERIFY TRAINING IS HAPPENING
   └─ Save model after 1 epoch
   └─ Does loss decrease?
   └─ Do outputs change from random?

4. CHECK EVALUATION THRESHOLD
   └─ Current: peaks detected at > 0.2 confidence
   └─ Maybe threshold is too strict?
   └─ Try different thresholds (0.1, 0.15, 0.5, etc.)

5. VERIFY NO DATA LEAKAGE/MISMATCH
   └─ Training set: 70% of images
   └─ Validation set: 15% of images
   └─ Test set: 15% of images
   └─ Do sets actually separate by image ID?
"""

    print(diagnosis)


if __name__ == "__main__":
    analyze_model_outputs()
