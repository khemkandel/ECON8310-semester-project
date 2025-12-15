# Baseball Moving Object Detection

A deep learning pipeline for detecting and classifying moving vs. non-moving objects in baseball video frames using **PyTorch** and **Faster R-CNN**.

## Overview

This project implements an object detection system trained on annotated baseball video frames. The model distinguishes between:
- **Moving objects** (e.g., baseballs in motion, players running)
- **Non-moving objects** (e.g., stationary players, equipment)

The system uses a **Faster R-CNN** architecture with a **ResNet50 + FPN** backbone, fine-tuned for 3-class classification (background, non-moving, moving).

## Features

- **Video Frame Extraction**: Automatically extract frames from `.mov` video files
- **CVAT XML Annotation Support**: Parse bounding box annotations from CVAT XML format
- **Transfer Learning**: Leverages pre-trained COCO weights for faster convergence
- **Train/Test Pipeline**: 80/20 split with automatic evaluation and visualization
- **Video Inference**: Process new videos and output annotated results

## Project Structure

```
├── main.py                 # Entry point - runs training and evaluation
├── config.py               # Configuration parameters
├── data_loader.py          # Dataset class for loading frames and annotations
├── train_eval.py           # Training loop, evaluation, and visualization
├── inference_video.py      # Run inference on new videos
├── ARCHITECTURE.md         # Detailed architecture documentation
├── TRAIN_EVAL_OVERVIEW.md  # Training pipeline overview
├── annotations/            # CVAT XML annotation files
├── frames/                 # Extracted video frames (organized by video)
├── videos/                 # Source video files
├── model/                  # Saved model checkpoints
├── new_videos/             # Input folder for video inference
└── output_videos/          # Processed videos with detections
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- Pillow

### Installation

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib pillow
```

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | (224, 224) | Input image dimensions |
| `MODEL_NUM_CLASSES` | 3 | Classes: background, non-moving, moving |
| `MODEL_LEARNING_RATE` | 1e-4 | Learning rate for AdamW optimizer |
| `NUM_EPOCHS` | 5 | Training epochs |
| `BATCH_SIZE` | 8 | Batch size for DataLoader |
| `TRAIN_RATIO` | 0.8 | Train/test split ratio |
| `CONFIDENCE_THRESHOLD` | 0.3 | Minimum confidence for predictions |
| `USE_TRAINED_MODEL` | True | Load pre-trained weights or train fresh |

## Usage

### Training & Evaluation

Run the main training and evaluation pipeline:

```bash
python main.py
```

This will:
1. Load the dataset (frames + XML annotations)
2. Split into 80% training / 20% test
3. Train the Faster R-CNN model (if `USE_TRAINED_MODEL=False`)
4. Evaluate on the test set
5. Generate visualizations with predictions

### Video Inference

To run inference on new videos:

1. Place your `.mov` video files in the `new_videos/` folder
2. Run the inference script:

```bash
python inference_video.py
```

3. Find processed videos with bounding box overlays in `output_videos/`

## Model Architecture

The system uses **Faster R-CNN** with the following configuration:

- **Backbone**: ResNet50 with Feature Pyramid Network (FPN)
- **Pre-trained weights**: COCO dataset (91 classes)
- **Custom head**: FastRCNNPredictor modified for 3 classes
- **Optimizer**: AdamW

### Class Labels

| Label | Class |
|-------|-------|
| 0 | Background |
| 1 | Non-moving object |
| 2 | Moving object |

## Data Format

### Annotations (CVAT XML)

The project uses CVAT XML format for annotations. Each XML file contains:
- Track IDs for each object
- Bounding box coordinates (`xtl`, `ytl`, `xbr`, `ybr`)
- `moving` attribute (true/false)

### Directory Structure

Frames should be organized as:
```
frames/
├── video_name_1/
│   ├── video_name_1_frame0.jpg
│   ├── video_name_1_frame1.jpg
│   └── ...
└── video_name_2/
    └── ...
```

## Output

### Training Output

- Model checkpoints saved to `model/` after each epoch
- Final model: `model/fasterrcnn_moving_objects.pth`

### Visualization

During evaluation, the system generates:
- Side-by-side comparisons of ground truth vs. predictions
- Color-coded bounding boxes:
  - **Purple**: Moving objects
  - **Green**: Non-moving objects (ground truth)
  - **Red**: Non-moving objects (predictions)

## GPU Support

The system automatically detects and uses CUDA if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## License

This project was developed for ECON 8310 semester project.

## Acknowledgments

- PyTorch and torchvision teams for the Faster R-CNN implementation
- CVAT for the annotation tool and XML format
