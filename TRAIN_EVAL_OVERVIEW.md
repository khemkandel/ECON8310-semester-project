# train_eval.py - Overview

```
                     INPUT: Video Frames + Annotations
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  BaseballData        │
                        │  (data_loader.py)    │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              ┌─────▼──────┐            ┌────────▼────┐
              │ TRAINING   │            │ TESTING     │
              │ 80% Data   │            │ 20% Data    │
              └─────┬──────┘            └────────┬────┘
                    │                           │
              ┌─────▼────────────┐     ┌────────▼────────┐
              │ Forward Pass +   │     │ Run Predictions │
              │ Backpropagation  │     │ Get Boxes/      │
              │ (AdamW)          │     │ Scores          │
              └─────┬────────────┘     └────────┬────────┘
                    │                           │
              ┌─────▼──────────┐       ┌────────▼────────┐
              │ Save Checkpoint│       │ Output Report   │
              │ (5 epochs)     │       │ + Visualizations│
              └────────────────┘       └─────────────────┘

    Framework: PyTorch | Model: Faster R-CNN | Classes: 3
```  