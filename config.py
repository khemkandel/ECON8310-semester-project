"""
Configuration file for baseball moving object detection project.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
import os
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = 'videos'
ANNOTATION_FOLDER = 'annotations'
EXTRACT_VIDEOS = False
IMAGE_SIZE = (224, 224)

# Data split ratio
TRAIN_RATIO = 0.3  # 30% training, 70% testing

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NUM_CLASSES = 3  # 0=background, 1=non-moving, 2=moving
MODEL_LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 8

# Model paths
MODEL_DIR = "model"
TRAINED_MODEL_PATH = "model/fasterrcnn_moving_objects.pth"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
USE_TRAINED_MODEL = True  # Set to True to load trained weights instead of training
CONFIDENCE_THRESHOLD = 0.3

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
NUM_MOVING_IMAGES_TO_DISPLAY = 10
NUM_NON_MOVING_IMAGES_TO_DISPLAY = 1

# Color scheme for visualization
COLOR_GT_MOVING = 'purple'
COLOR_GT_NON_MOVING = 'green'
COLOR_PRED_MOVING = 'purple'
COLOR_PRED_NON_MOVING = 'red'

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Device will be automatically detected (CUDA if available, CPU otherwise)