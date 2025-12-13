# Baseball Moving Object Detection - Executive Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BASEBALL OBJECT DETECTION PIPELINE                 │
│                     (PyTorch + Faster R-CNN)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA LOADING PIPELINE                           │
└──────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │   BaseballData (Dataset)│
                    │  - Load XML annotations │
                    │  - Extract video frames │
                    │  - Resize to 224×224    │
                    │  - Scale bboxes         │
                    └────────┬────────────────┘
                             │
                    ┌────────▼──────────┐
                    │  random_split()   │
                    │  80% / 20%        │
                    └──────┬───────┬────┘
                           │       │
          ┌────────────────┘       └──────────────────┐
          │                                           │
    ┌─────▼──────────┐                        ┌──────▼──────┐
    │ Train Dataset  │                        │ Test Dataset│
    │ (Subset Objs)  │                        │(Subset Objs)│
    └─────┬──────────┘                        └──────┬──────┘
          │                                         │
    ┌─────▼──────────────┐               ┌─────────▼────────┐
    │  DataLoader (Batch)│               │ DataLoader (Batch)│
    │  batch_size = 8    │               │  batch_size = 8   │
    │  collate_fn()      │               │  collate_fn()     │
    └─────┬──────────────┘               └─────────┬────────┘
          │                                        │
          └────────────┬─────────────────────────┘
                       │
              ┌────────▼─────────┐
              │  Tensor Batches   │
              │  - Images (GPU)   │
              │  - Labels (GPU)   │
              │  - BBoxes (GPU)   │
              └───────────────────┘
```

---

## 2. Model Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      FASTER R-CNN ARCHITECTURE                           │
│                    (Pre-trained ResNet50 + FPN)                          │
└──────────────────────────────────────────────────────────────────────────┘

  INPUT: Image (3, 224, 224)
    │
    ▼
  ┌─────────────────────────────────────┐
  │    BACKBONE: ResNet50 + FPN         │
  │  (Feature Pyramid Network)          │
  │  Pre-trained on COCO (91 classes)   │
  └────────┬────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────┐
  │   RPN (Region Proposal Network)     │
  │  Generate candidate bounding boxes  │
  │  Loss: objectness + box regression  │
  └────────┬────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────┐
  │   ROI HEAD (Region of Interest)     │
  │  1. Extract features for each ROI   │
  │  2. Classification head             │
  │  3. Bounding box regression         │
  └────────┬────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────┐
  │  CUSTOM FastRCNNPredictor Head           │
  │  ┌────────────────────────────────────┐  │
  │  │ Input Features: 1024               │  │
  │  │ Output Classes: 3                  │  │
  │  │  - Class 0: Background             │  │
  │  │  - Class 1: Non-moving object      │  │
  │  │  - Class 2: Moving object          │  │
  │  └────────────────────────────────────┘  │
  └────────┬─────────────────────────────────┘
           │
           ▼
  OUTPUT: 4 Loss Components (Training) / Detections (Inference)
  ┌─────────────────────────────────┐
  │ Loss Dictionary {               │
  │   classifier_loss               │
  │   box_reg_loss                  │
  │   objectness_loss               │
  │   rpn_box_reg_loss              │
  │ }                               │
  └─────────────────────────────────┘
```

---

## 3. Training Loop

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE                                  │
│                    (NUM_EPOCHS = 5)                                      │
└──────────────────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌─────────────────────────────────────┐
│ FOR each epoch (1 to NUM_EPOCHS)    │
└────────┬────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │ FOR each batch in train_loader  │
    └────────┬────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────┐
    │ 1. Move to GPU/CPU device             │
    │    - Images to device                  │
    │    - Labels to device (adjust 0→1, 1→2)│
    │    - BBoxes to device                  │
    └────────┬─────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ 2. FORWARD PASS                      │
    │    model(images, targets)             │
    │    ↓                                  │
    │    Returns loss_dict with 4 losses   │
    └────────┬──────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ 3. COMPUTE TOTAL LOSS                │
    │    loss = sum(loss_dict.values())     │
    │    Single scalar for backpropagation  │
    └────────┬──────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ 4. BACKWARD PASS                     │
    │    optimizer.zero_grad()              │
    │    loss.backward()  ← Compute grads  │
    │    optimizer.step() ← Update weights  │
    └────────┬──────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ 5. ACCUMULATE METRICS                │
    │    epoch_loss += loss.item()          │
    │    Print batch loss                   │
    └────────┬──────────────────────────────┘
             │
             └─→ (next batch)
                     │
                     ▼ (all batches done)
         ┌──────────────────────────────────┐
         │ 6. SAVE MODEL CHECKPOINT         │
         │    torch.save(model.state_dict())│
         │    → model/epoch{N}.pth          │
         └──────────────────────────────────┘
                     │
                     └─→ (next epoch)

END (After NUM_EPOCHS)
```

---

## 4. Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION PIPELINE                                │
│                     (Test Set Assessment)                                │
└──────────────────────────────────────────────────────────────────────────┘

START: model.eval() + torch.no_grad()
  │
  ▼
┌─────────────────────────────────────┐
│ FOR each batch in test_loader       │
└────────┬────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 1. INFERENCE (No Gradients)              │
│    outputs = model(images)                │
│    Returns detections with:               │
│    - boxes (coordinates)                  │
│    - scores (confidence [0,1])            │
│    - labels (class predictions)           │
└────────┬──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ 2. FILTER BY CONFIDENCE THRESHOLD (0.3)     │
│    keep = scores >= CONFIDENCE_THRESHOLD     │
│    boxes_filtered = boxes[keep]              │
│    scores_filtered = scores[keep]            │
│    labels_filtered = labels[keep]            │
└────────┬──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 3. LABEL ADJUSTMENT (Reverse training)  │
│    labels_filtered = labels_filtered - 1 │
│    (Convert 1,2 → 0,1)                   │
└────────┬──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 4. COMPUTE GROUND TRUTH STATISTICS      │
│    gt_moving = count(labels == 1)        │
│    gt_non_moving = count(labels == 0)    │
│    Accumulate totals across all images   │
└────────┬──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 5. TRACK PREDICTION METRICS              │
│    total_predictions += num_detections   │
│    total_confidences.extend(scores)      │
└────────┬──────────────────────────────────┘
         │
         └─→ (next batch)
                 │
                 ▼ (all batches done)
    ┌────────────────────────────────────────┐
    │ 6. PRINT EVALUATION SUMMARY            │
    │    - Total GT objects                  │
    │    - Moving vs Non-moving breakdown    │
    │    - Total predictions                 │
    │    - Avg/Min/Max confidence scores     │
    └────────────────────────────────────────┘

END
```

---

## 5. Visualization Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION PIPELINE                                │
│              (Side-by-side GT vs Predictions)                            │
└──────────────────────────────────────────────────────────────────────────┘

INPUT: Moving/Non-moving image indices (pre-categorized)
  │
  ▼
┌──────────────────────────────────────┐
│ 1. PRE-CATEGORIZE TEST SET           │
│    Scan all test images               │
│    - has_moving = any(labels == 1)   │
│    - has_non_moving = any(labels==0) │
│    Build indices for each category    │
└────────┬──────────────────────────────┘
         │
         ▼
    ┌──────────────────────────┐
    │ FOR each display image   │
    └────────┬─────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────┐
    │ 2. RETRIEVE PREDICTIONS FOR IMAGE       │
    │    - Filter by confidence (≥0.3)        │
    │    - Adjust labels (1,2 → 0,1)          │
    │    - Extract coordinates & scores       │
    └────────┬────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────┐
    │ 3. CREATE SIDE-BY-SIDE VISUALIZATION      │
    │                                            │
    │  ┌──────────────────┐  ┌──────────────────┐
    │  │  LEFT: GT BOXES  │  │ RIGHT: PRED BOXES│
    │  │                  │  │                  │
    │  │ Green: Non-mov   │  │ Red: Non-mov     │
    │  │ Purple: Moving   │  │ Purple: Moving   │
    │  │ Confidence: —    │  │ Confidence: 0.92 │
    │  │                  │  │                  │
    │  └──────────────────┘  └──────────────────┘
    │                                            │
    │  Title: Image type + object counts        │
    └────────┬─────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ 4. DISPLAY STATISTICS                │
    │    - GT: 2 non-moving, 3 moving      │
    │    - Pred: 2 non-moving, 3 moving    │
    │    - Avg confidence: 0.87             │
    └──────────────────────────────────────┘
             │
             └─→ (next image)

END
```

---

## 6. Configuration System

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       CONFIG.PY PARAMETERS                               │
└──────────────────────────────────────────────────────────────────────────┘

CONFIG
├── DATA CONFIGURATION
│   ├── BASE_FOLDER = config.py directory (dynamic)
│   ├── VIDEO_FOLDER = 'videos'
│   ├── ANNOTATION_FOLDER = 'annotations'
│   ├── EXTRACT_VIDEOS = False
│   ├── IMAGE_SIZE = (224, 224)
│   └── TRAIN_RATIO = 0.3
│
├── MODEL CONFIGURATION
│   ├── MODEL_NUM_CLASSES = 3
│   ├── MODEL_LEARNING_RATE = 1e-4
│   ├── NUM_EPOCHS = 5
│   ├── BATCH_SIZE = 8
│   ├── MODEL_DIR = "model"
│   └── TRAINED_MODEL_PATH = "model/fasterrcnn_moving_objects.pth"
│
├── TRAINING CONFIGURATION
│   ├── USE_TRAINED_MODEL = True
│   └── CONFIDENCE_THRESHOLD = 0.3
│
└── VISUALIZATION CONFIGURATION
    ├── NUM_MOVING_IMAGES_TO_DISPLAY = 10
    ├── NUM_NON_MOVING_IMAGES_TO_DISPLAY = 1
    ├── COLOR_GT_MOVING = 'purple'
    ├── COLOR_GT_NON_MOVING = 'green'
    ├── COLOR_PRED_MOVING = 'purple'
    └── COLOR_PRED_NON_MOVING = 'red'
```

---

## 7. File Structure & Dependencies

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       PROJECT FILE HIERARCHY                             │
└──────────────────────────────────────────────────────────────────────────┘

ECON8310-semester-project/
│
├── main.py ────────────────────────┐
│   (Entry Point)                   │
│   └──> Calls train_eval.main()   │
│                                  │
├── config.py ◄─────────────────────┴──────┐
│   (Settings & Parameters)                  │
│   └──> Imported by all modules            │
│                                           │
├── data_loader.py ◄────────────────────────┤
│   (BaseballData Dataset Class)            │
│   └──> Returns (images, labels, bboxes)  │
│                                           │
├── train_eval.py ◄────────────────────────┤
│   (BaseballModelTrainer Class)            │
│   ├──> _load_data() → DataLoaders        │
│   ├──> _initialize_model() → Faster RCNN │
│   ├──> train() → Training loop            │
│   ├──> evaluate() → Test metrics          │
│   ├──> _precategorize_test_set() → indices
│   ├──> visualize_samples() → Plots       │
│   └──> save_model() → Checkpoint          │
│                                           │
├── annotations/ ◄─────────────────────────┤
│   └──> XML files with bounding boxes     │
│                                           │
├── frames/ ◄───────────────────────────────┤
│   └──> Image directories per video       │
│                                           │
├── videos/ ◄───────────────────────────────┤
│   └──> Video files (extracted if needed)  │
│                                           │
└── model/ ◄────────────────────────────────┘
    └──> PyTorch checkpoints (*.pth)
```

---

## 8. Data Label Transformation

```
┌──────────────────────────────────────────────────────────────────────────┐
│               LABEL ADJUSTMENT FOR FASTER R-CNN                          │
│         (Faster R-CNN reserves class 0 for background)                   │
└──────────────────────────────────────────────────────────────────────────┘

INPUT LABELS (from annotations)     INTERNAL LABELS (Faster R-CNN)
                                    
    0 = Non-moving object    ──→      1 = Non-moving object
    1 = Moving object        ──→      2 = Moving object
    -                        ──→      0 = Background (reserved)

TRAINING:
  Input: labels[0, 1]  →  adjusted = labels + 1  →  [1, 2]
  Pass to model (expects 0 for background)

INFERENCE:
  Output: labels[1, 2]  →  adjusted = labels - 1  →  [0, 1]
  Convert back to original format
```

---

## 9. Key Metrics & Outputs

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING OUTPUTS                                  │
└──────────────────────────────────────────────────────────────────────────┘

DURING TRAINING:
  ✓ Batch loss printed per batch
  ✓ Epoch loss printed after all batches
  ✓ Model checkpoint saved per epoch
    → model/fasterrcnn_moving_objects_epoch1.pth
    → model/fasterrcnn_moving_objects_epoch2.pth
    → ... (5 total)

DURING EVALUATION:
  ✓ Total ground truth objects (moving vs non-moving)
  ✓ Total high-confidence predictions
  ✓ Average prediction confidence score
  ✓ Min/Max confidence range

VISUALIZATION OUTPUT:
  ✓ Side-by-side GT vs Predictions
  ✓ Color-coded bounding boxes
  ✓ Object counts per category
  ✓ Confidence scores on predictions
```

---

## 10. Device Management

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    GPU/CPU DEVICE MANAGEMENT                             │
└──────────────────────────────────────────────────────────────────────────┘

DEVICE SELECTION:
  device = "cuda" if torch.cuda.is_available() else "cpu"

TENSOR MOVEMENT:
  Training:    images.to(device)  ← Keep on GPU during training
  Inference:   .detach().cpu()    ← Move to CPU after inference
               (More efficient for post-processing & visualization)

MODEL PLACEMENT:
  model.to(device)  ← Entire model moved to device
```

---

## Quick Start Command

```bash
python main.py
```

This executes:
1. Initialize trainer + load data
2. Pre-categorize test set
3. Train for NUM_EPOCHS
4. Evaluate on test set
5. Visualize results
6. Save final model

---

**Created: December 11, 2025**
**Project: Baseball Moving Object Detection (ECON 8310)**
