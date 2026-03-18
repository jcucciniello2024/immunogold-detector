"""Quick evaluation script to get accuracy metrics from best model."""
import os
import sys
import csv
import numpy as np
import torch
import tifffile
from scipy import ndimage

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from dataset_points import _to_chw_01
from model_unet_deep import UNetDeepKeypointDetector
from prepare_labels import discover_image_records

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load best model (epoch 32, lowest val loss, anti-overfit)
model = UNetDeepKeypointDetector(in_channels=3, out_channels=2, base_channels=32).to(device)
model_path = "checkpoints/4594628/detector_best.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model from {model_path}")

# Discover test records
records = discover_image_records("data/Max Planck Data/Gold Particle Labelling/analyzed synapses")
print(f"Discovered {len(records)} images")

# Simple train/val/test split
np.random.seed(42)
idx = np.arange(len(records))
np.random.shuffle(idx)
n_train = max(1, int(round(len(records) * 0.7)))
n_val = max(1, int(round(len(records) * 0.15)))
test_idx = idx[n_train + n_val:]
test_records = [records[i] for i in test_idx]
print(f"Evaluating on {len(test_records)} test images")

# Generate predictions on test set
predictions = []
with torch.no_grad():
    for rec in test_records:
        try:
            img = _to_chw_01(tifffile.imread(rec.image_path))
            c, h, w = img.shape
            
            # Forward pass
            img_t = torch.from_numpy(img).float().unsqueeze(0).to(device)
            logits = model(img_t)
            probs = torch.sigmoid(logits)  # (1, 2, H, W)
            
            # Extract peaks per class (6nm and 12nm)
            for cls_id in [0, 1]:
                hm = probs[0, cls_id].cpu().numpy()
                
                # Simple peak detection via local maxima
                local_max = ndimage.maximum_filter(hm, size=5) == hm
                peaks = np.argwhere(local_max & (hm > 0.3))
                
                for y, x in peaks:
                    conf = float(hm[y, x])
                    predictions.append({
                        'image_id': rec.image_id,
                        'class_id': cls_id,
                        'x': float(x),
                        'y': float(y),
                        'confidence': conf
                    })
        except Exception as e:
            print(f"Error processing {rec.image_id}: {e}")

# Save predictions CSV
pred_csv = "predictions_best.csv"
with open(pred_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_id', 'class_id', 'x', 'y', 'confidence'])
    writer.writeheader()
    writer.writerows(predictions)

print(f"Saved {len(predictions)} predictions to {pred_csv}")
print("\nTo evaluate:")
print(f"python evaluate_detector.py --data_root data/Max --pred_csv {pred_csv}")
