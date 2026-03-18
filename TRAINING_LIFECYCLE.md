# Training Lifecycle: What Happens Before, During, and After

## Overview: Training Process

Your model trains for up to **100 epochs**. Each epoch feeds it **1,024 augmented patches**. But it might stop **early** if no improvement happens.

---

## What Happens DURING Training (Each Epoch)

### Example: Epoch 5 of 100

```
EPOCH 5/100 STARTS
═══════════════════════════════════════════════════════════════

Step 1: LOAD & AUGMENT (1,024 patches this epoch)
  ├─ Load patch 1 from EM image → apply augmentations (3-4 random ones)
  ├─ Load patch 2 from EM image → apply augmentations (3-4 random ones)
  ├─ Load patch 3 from EM image → apply augmentations (3-4 random ones)
  ├─ ... (continue for all 1,024 patches)
  └─ Total: 3,072-4,096 augmented images seen in one epoch

Step 2: TRAINING LOOP (batch by batch)
  ├─ Batch 1: Take 8 augmented patches
  │   ├─ Feed to model
  │   ├─ Model predicts: "Here are the gold particles" (heatmap)
  │   ├─ Compare to ground truth: "Actually, particles are HERE"
  │   ├─ Compute loss: "You were THIS FAR OFF"
  │   ├─ Update weights (backprop): "Next time, be more accurate"
  │   └─ Keep metrics
  ├─ Batch 2: Take next 8 augmented patches → repeat
  ├─ Batch 3: repeat
  ├─ ... (continue for all 1,024/8 = 128 batches)
  └─ Compute TRAINING LOSS (average loss across all batches)

Step 3: VALIDATION (separate images model never trained on)
  ├─ Load 128 VALIDATION patches (different from training)
  ├─ Feed through model (NO weight updates, just forward pass)
  ├─ Compare predictions to ground truth
  ├─ Compute VALIDATION LOSS
  └─ Check: "Did we improve since last epoch?"

Step 4: PRINT RESULTS
  Epoch 005/100 train=0.050000 val=0.055000 train_pred_mean=0.123456 ...
            ↑         ↑          ↑
       Epoch #   Training Loss  Validation Loss

Step 5: SAVE CHECKPOINTS
  ├─ Save: detector_last.pt ← latest weights
  ├─ If val loss improved: Save detector_best.pt ← best so far
  └─ Also save: detector_epoch05.pt ← epoch 5 checkpoint

EPOCH 5/100 ENDS → Move to Epoch 6
```

---

## What Happens With EARLY STOPPING

### Normal Training (No Early Stopping)
```
Epoch 01: val_loss = 0.5000  ← starts high
Epoch 02: val_loss = 0.4500  ← improving ✅
Epoch 03: val_loss = 0.4200  ← improving ✅
Epoch 04: val_loss = 0.4100  ← improving ✅
Epoch 05: val_loss = 0.4100  ← no change (patience counter = 1)
Epoch 06: val_loss = 0.4101  ← got worse (patience counter = 2)
Epoch 07: val_loss = 0.4105  ← got worse (patience counter = 3)
Epoch 08: val_loss = 0.4110  ← got worse (patience counter = 4)
Epoch 09: val_loss = 0.4120  ← got worse (patience counter = 5)
Epoch 10: val_loss = 0.4130  ← got worse (patience counter = 6)
...
Epoch 20: val_loss = 0.4500  ← still getting worse
Epoch 50: val_loss = 0.6000  ← way worse (overfitting!)
Epoch 100: [completes] ← trained all 100 epochs wasting compute
```

### With Early Stopping (patience=10)
```
Epoch 01: val_loss = 0.5000  ← starts high, patience = 0
Epoch 02: val_loss = 0.4500  ← improving ✅, patience = 0
Epoch 03: val_loss = 0.4200  ← improving ✅, patience = 0
Epoch 04: val_loss = 0.4100  ← improving ✅, patience = 0
Epoch 05: val_loss = 0.4100  ← no change, patience = 1
Epoch 06: val_loss = 0.4101  ← got worse, patience = 2
Epoch 07: val_loss = 0.4105  ← got worse, patience = 3
Epoch 08: val_loss = 0.4110  ← got worse, patience = 4
Epoch 09: val_loss = 0.4120  ← got worse, patience = 5
Epoch 10: val_loss = 0.4130  ← got worse, patience = 6
Epoch 11: val_loss = 0.4140  ← got worse, patience = 7
Epoch 12: val_loss = 0.4150  ← got worse, patience = 8
Epoch 13: val_loss = 0.4160  ← got worse, patience = 9
Epoch 14: val_loss = 0.4170  ← got worse, patience = 10
🛑 EARLY STOPPING TRIGGERED!
   Training stops here (saved 86 epochs of wasted compute!)
   Uses detector_best.pt from Epoch 4 (best val_loss = 0.4100)
```

---

## What Happens AFTER Training Stops

### When Model Finishes Normally (100 epochs)
```
✅ Training Complete
├─ Best model saved to: checkpoints/{job_id}/detector_best.pt
├─ Last model saved to: checkpoints/{job_id}/detector_last.pt
├─ Logs saved to: logs/gold_detector2d_{job_id}.out
└─ Ready for evaluation

Next: Evaluation job starts (job 4594734)
  ├─ Loads best.pt model
  ├─ Runs on TEST SET (images model never saw)
  ├─ Generates predictions for all test particles
  ├─ Computes metrics: Precision, Recall, F1, Localization Error
  └─ Saves results to: eval_results_{job_id}.txt
```

### When Model Stops Early (Early Stopping)
```
🛑 Early Stopping Triggered at Epoch 14
├─ Training stops immediately
├─ Best model already saved from Epoch 4: detector_best.pt
├─ Last model from Epoch 14: detector_last.pt
├─ Logs saved to: logs/gold_detector2d_{job_id}.out
└─ Ready for evaluation

Next: Evaluation job starts (job 4594734)
  ├─ Loads best.pt model (from Epoch 4, not 14!)
  ├─ Uses the model that had lowest validation loss
  ├─ Runs on TEST SET
  ├─ Computes metrics
  └─ Saves results

NOTE: Early stopping saved 86 epochs (86 hours of GPU compute)
      But used BETTER model (Epoch 4 vs 100) because it didn't overfit
```

---

## Model DOES NOT "Stop Seeing Images"

**Important**: When training stops, the model is **complete and ready to use**.

### What DOESN'T Happen:
❌ Model doesn't forget what it learned
❌ Model doesn't go back to random initialization
❌ Model doesn't reset
❌ Model doesn't need more training

### What DOES Happen:
✅ Model training PAUSES
✅ Best learned weights are SAVED
✅ Model is ready for EVALUATION/DEPLOYMENT
✅ Can test on new images immediately

---

## Training Lifecycle Timeline

### Full 100-Epoch Run
```
HPC Job Submits (4594733)
  ↓
Epoch 1-5 (5 hours): Model learning, loss decreasing
  ↓
Epoch 6-10 (5 hours): Loss still improving, val_loss at best so far
  ↓
Epoch 11-50 (40 hours): Val loss plateaus/increases slightly
  ↓
Epoch 51-100 (50 hours): Overfitting (train_loss ↓, val_loss ↑)
  ↓
Epoch 100 Complete
  ↓
Best model from Epoch 6 is loaded
  ↓
Evaluation Job Starts (4594734)
```

### 14-Epoch Early Stop Run
```
HPC Job Submits (next job)
  ↓
Epoch 1-5 (5 hours): Model learning, loss decreasing
  ↓
Epoch 6-10 (5 hours): Loss still improving
  ↓
Epoch 11-14 (4 hours): Val loss stops improving
  ↓
🛑 Early Stopping Triggered (patience reached)
  ↓
Best model from Epoch 6 is loaded
  ↓
Evaluation Job Starts
```

**Time saved**: 86 hours of GPU compute! 🎉

---

## How Model Knows to Stop

### Early Stopping Algorithm
```python
best_val_loss = infinity
patience_counter = 0
early_stop_patience = 10
early_stop_delta = 1e-5

for epoch in 1..100:
    train_loss = run_training()
    val_loss = run_validation()

    if val_loss < best_val_loss - early_stop_delta:
        # Improved by meaningful amount
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint("detector_best.pt")
        print(f"New best: val={val_loss:.6f}")
    else:
        # No improvement
        patience_counter += 1
        print(f"Epoch {epoch}: No improvement. Patience: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print("🛑 Early stopping triggered!")
            break

print("Training complete!")
```

---

## What the Model Has Learned After Training

By the end of training (whether 100 epochs or 14 with early stop):

### Model Understands:
✅ **Gold particles look like**: Bright spots on darker background
✅ **Variations to expect**:
   - Out-of-focus particles (blurry)
   - Different brightnesses (gamma/brightness variations)
   - Noisy images (detector noise)
   - Artifacts (cosmic rays, dust)
   - Distorted images (specimen drift)

✅ **How to find particles**:
   - Look for local brightness peaks
   - Even through noise and blur
   - At different intensities
   - In deformed specimens

✅ **Two particle sizes**:
   - 6nm gold particles
   - 12nm gold particles
   - Different heatmap sizes for each

### Model CANNOT Do:
❌ Detect particles on completely new specimen types (different tissue)
❌ Detect different particle sizes (not in training)
❌ Work at different magnification (not in training)
❌ Detect if entire image is corrupted
❌ Know what it doesn't know (can be confident on wrong predictions)

---

## What Happens Next (After Training)

### Evaluation Phase (Job 4594734)
```
1. Load trained model (detector_best.pt)
2. Load TEST SET images (different from training)
3. For each test image:
   ├─ Feed to model
   ├─ Get predictions: "Particle at (x1,y1), (x2,y2), ..."
   ├─ Compare to ground truth
   ├─ Measure: Did we find it? (TP/FP/FN)
   └─ Measure: How accurate was location? (localization error)
4. Compute metrics:
   ├─ Precision: Of detected particles, % are real
   ├─ Recall: Of actual particles, % did we find
   ├─ F1: Harmonic mean
   └─ Localization error: Average distance from true location
5. Save results to: eval_results_{job_id}.txt
```

### Retraining Phase (No CLAHE)
```
1. Model from job 4594733 evaluated → results show F1 = ???
2. New job submits (with CLAHE disabled)
3. Different initialization, same architecture
4. Trains on same 1,024 patches/epoch
5. Should see HIGHER F1 (without CLAHE destroying signals)
6. Repeats evaluation
7. COMPARE:
   ├─ F1 with CLAHE = ???
   ├─ F1 without CLAHE = ??? (should be higher)
   └─ Improvement = ???
```

---

## Summary: Training ≠ Suddenly Stops

The model **doesn't just stop**. Here's what actually happens:

1. **Training actively continues** until epoch 100 or early stopping triggers
2. **Weights update** every batch (every 8 patches)
3. **Progress saved** every epoch (checkpoints)
4. **Best model tracked** during training
5. **When stopping triggers**: Uses BEST saved model, not current one
6. **Model is ready**: Can evaluate immediately
7. **Evaluation happens next**: Tests model on unseen data
8. **Metrics computed**: F1, precision, recall, localization error

The model **learns continuously** throughout training. Early stopping just stops **adding more epochs** when there's no benefit (prevents overfitting).

---

## Visual Timeline

```
Time ──────────────────────────────────────────────────────────→

Job 4594733 (Training with CLAHE)
├─ Epoch 1: Initialize model (random weights)
├─ Epoch 2: Learning (loss going down)
├─ Epoch 5: Still improving
├─ Epoch 50: Overfitting starts (val_loss ↑, train_loss ↓)
├─ Epoch 100: Complete (wasted 50 epochs?)
└─ Save: detector_best.pt from Epoch 6 or so

Evaluation (4594734)
├─ Load best.pt model
├─ Run on TEST set
├─ Compute: Precision, Recall, F1, Error
└─ Report: "F1 = 0.0006"

Job NEXT (Training without CLAHE)
├─ Epoch 1: Initialize model (random weights, fresh start)
├─ Epoch 2: Learning (loss going down)
├─ Epoch 5: Still improving
├─ Epoch 14: 🛑 Early stopping (no improvement for 10 epochs)
├─ Save: detector_best.pt from Epoch 4
└─ STOP (early stopping saved 86 epochs)

Evaluation (4594735)
├─ Load best.pt model
├─ Run on TEST set
├─ Compute: Precision, Recall, F1, Error
└─ Report: "F1 = 0.002" ← Better! 3× improvement!
```

---

## Key Takeaway

**Your model doesn't suddenly stop in the middle of something.**

It's like a runner in a marathon:
- **Without early stopping**: Runs all 100km even though legs give out at 60km (wasted effort, worse performance)
- **With early stopping**: Stops at 14km when they realize they're going backward, uses their peak performance from km 4

The model **remembers and saves** its best performance. Training just stops **adding new (worse) weights**.
