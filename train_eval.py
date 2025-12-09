"""
Training and evaluation pipeline for baseball moving object detection.
Handles model initialization, training loop, evaluation, and visualization.
"""

import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split

from config import (
    BASE_FOLDER, VIDEO_FOLDER, ANNOTATION_FOLDER, IMAGE_SIZE,
    TRAIN_RATIO, MODEL_NUM_CLASSES, MODEL_LEARNING_RATE, NUM_EPOCHS,
    BATCH_SIZE, MODEL_DIR, TRAINED_MODEL_PATH, USE_TRAINED_MODEL,
    CONFIDENCE_THRESHOLD, NUM_MOVING_IMAGES_TO_DISPLAY, NUM_NON_MOVING_IMAGES_TO_DISPLAY,
    COLOR_GT_MOVING, COLOR_GT_NON_MOVING, COLOR_PRED_MOVING, COLOR_PRED_NON_MOVING, EXTRACT_VIDEOS
)
from data_loader import BaseballData


class BaseballModelTrainer:
    """Handles model training, evaluation, and visualization."""

    def __init__(self, device=None):
        """Initialize trainer with device and configuration."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Load dataset
        self._load_data()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=MODEL_LEARNING_RATE)

    def _load_data(self):
        """Load and prepare dataset."""
        print("\nLoading dataset...")
        full_dataset = BaseballData(
            base_folder=BASE_FOLDER,
            videofolder=VIDEO_FOLDER,
            annotation_folder=ANNOTATION_FOLDER,
            extract_videos=EXTRACT_VIDEOS,
            image_size=IMAGE_SIZE
        )
        
        # Split dataset
        train_size = int(TRAIN_RATIO * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda x: full_dataset.collate_fn(x)
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda x: full_dataset.collate_fn(x)
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

    def _initialize_model(self):
        """Initialize Faster R-CNN model with custom head."""
        print("\nInitializing model...")
        
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            num_classes=91
        )
        
        # Replace classification head for custom classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=MODEL_NUM_CLASSES)
        
        model.to(self.device)
        print(f"Model configured for {MODEL_NUM_CLASSES} classes: 0=background, 1=non-moving, 2=moving")
        
        # Load trained model if requested
        if USE_TRAINED_MODEL:
            if os.path.exists(TRAINED_MODEL_PATH):
                print(f"\nLoading trained model from: {TRAINED_MODEL_PATH}")
                model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=self.device))
                print("Trained model loaded successfully.")
                model.eval()
            else:
                print(f"\nWARNING: Trained model not found at {TRAINED_MODEL_PATH}")
                print("Proceeding with fresh model initialization.")
        else:
            model.train()
            print("\nUSE_TRAINED_MODEL is False. Training fresh model...")
        
        return model

    def _precategorize_test_set(self):
        """Categorize test images by object type for visualization."""
        print("\nPre-scanning test set to categorize images...")
        
        moving_object_indices = []
        non_moving_object_indices = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, bboxs) in enumerate(self.test_loader):
                for img_idx, (img_labels, img_boxes) in enumerate(zip(labels, bboxs)):
                    has_moving = (img_labels == 1).sum().item() > 0
                    has_non_moving = (img_labels == 0).sum().item() > 0
                    
                    if has_moving:
                        moving_object_indices.append((batch_idx, img_idx))
                    elif has_non_moving:
                        non_moving_object_indices.append((batch_idx, img_idx))
        
        print(f"Test set categorization:")
        print(f"  Images with MOVING objects: {len(moving_object_indices)}")
        print(f"  Images with NON-MOVING objects only: {len(non_moving_object_indices)}")
        if len(moving_object_indices) > 0:
            print(f"  Sample moving indices: {moving_object_indices[:3]}")
        if len(non_moving_object_indices) > 0:
            print(f"  Sample non-moving indices: {non_moving_object_indices[:3]}")
        
        return moving_object_indices, non_moving_object_indices

    def train(self):
        """Train model for specified epochs."""
        if USE_TRAINED_MODEL:
            print("\nSkipping training - using loaded trained model.")
            return
        
        self.model.train()
        
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            print(f"\n{'='*70}")
            print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}...")
            print(f"{'='*70}")

            for images, labels, bboxs in self.train_loader:
                images = [img.to(self.device) for img in images]

                targets = []
                for lbl, bbox in zip(labels, bboxs):
                    # Adjust labels: 0->1 (non-moving), 1->2 (moving)
                    adjusted_labels = lbl + 1
                    targets.append({
                        "boxes": bbox.to(self.device),
                        "labels": adjusted_labels.to(self.device)
                    })

                # Forward pass
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                print(f"  Batch loss: {loss.item():.4f}")
            
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] - Total Loss: {epoch_loss:.4f}")
            torch.save(self.model.state_dict(), f"{MODEL_DIR}/fasterrcnn_moving_objects_epoch{epoch+1}.pth")
            print(f"Model saved to: {MODEL_DIR}/fasterrcnn_moving_objects_epoch{epoch+1}.pth")

    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET")
        print("="*70)

        self.model.eval()
        total_predictions = 0
        total_confidences = []
        gt_moving_count = 0
        gt_non_moving_count = 0

        with torch.no_grad():
            for batch_idx, (images, labels, bboxs) in enumerate(self.test_loader):
                images_gpu = [img.to(self.device) for img in images]
                outputs = self.model(images_gpu)
                
                for img_idx, output in enumerate(outputs):
                    boxes = output["boxes"].detach().cpu()
                    scores = output["scores"].detach().cpu()
                    labels_pred = output["labels"].detach().cpu()
                    
                    # Filter by confidence threshold
                    keep = scores >= CONFIDENCE_THRESHOLD
                    boxes_filtered = boxes[keep]
                    scores_filtered = scores[keep]
                    labels_filtered = labels_pred[keep]
                    
                    # Convert labels back: 1->0 (non-moving), 2->1 (moving)
                    if len(labels_filtered) > 0:
                        labels_filtered = labels_filtered - 1
                    
                    # Ground truth
                    gt_labels = labels[img_idx]
                    gt_moving = (gt_labels == 1).sum().item()
                    gt_non_moving = (gt_labels == 0).sum().item()
                    gt_moving_count += gt_moving
                    gt_non_moving_count += gt_non_moving
                    
                    # Track predictions
                    num_detections = len(boxes_filtered)
                    total_predictions += num_detections
                    if len(scores_filtered) > 0:
                        total_confidences.extend(scores_filtered.tolist())

        print("Evaluation complete.")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\nTest Set Statistics:")
        print(f"  Total GT objects: {gt_moving_count + gt_non_moving_count}")
        print(f"    - Non-moving: {gt_non_moving_count}")
        print(f"    - Moving: {gt_moving_count}")
        print(f"\n  Total predictions (score >= {CONFIDENCE_THRESHOLD}): {total_predictions}")
        if len(total_confidences) > 0:
            avg_confidence = sum(total_confidences) / len(total_confidences)
            print(f"  Average confidence: {avg_confidence:.4f}")
            print(f"  Min/Max confidence: {min(total_confidences):.4f} / {max(total_confidences):.4f}")
        
        return gt_moving_count, gt_non_moving_count, total_predictions, total_confidences

    def visualize_samples(self, moving_indices, non_moving_indices):
        """Visualize sample images from moving and non-moving sets."""
        print("\n" + "="*70)
        print("VISUALIZATION - SAMPLE IMAGES FROM CATEGORIZED INDICES")
        print("="*70)

        # Display moving object images
        print("\n" + "-"*70)
        print(f"DISPLAYING IMAGES WITH MOVING OBJECTS (Total: {len(moving_indices)})")
        print("-"*70)
        num_moving_to_display = min(NUM_MOVING_IMAGES_TO_DISPLAY, len(moving_indices))
        for idx in range(num_moving_to_display):
            batch_idx, img_idx = moving_indices[idx]
            self._display_image(batch_idx, img_idx, "MOVING")

        # Display non-moving object images
        print("\n" + "-"*70)
        print(f"DISPLAYING IMAGES WITH NON-MOVING OBJECTS ONLY (Total: {len(non_moving_indices)})")
        print("-"*70)
        num_non_moving_to_display = min(NUM_NON_MOVING_IMAGES_TO_DISPLAY, len(non_moving_indices))
        for idx in range(num_non_moving_to_display):
            batch_idx, img_idx = non_moving_indices[idx]
            self._display_image(batch_idx, img_idx, "NON-MOVING")

    def _display_image(self, batch_idx, img_idx, image_type):
        """Display image with GT and predicted bounding boxes."""
        with torch.no_grad():
            for test_batch_idx, (images, labels, bboxs) in enumerate(self.test_loader):
                if test_batch_idx == batch_idx:
                    images_gpu = [img.to(self.device) for img in images]
                    outputs = self.model(images_gpu)
                    
                    output = outputs[img_idx]
                    boxes = output["boxes"].detach().cpu()
                    scores = output["scores"].detach().cpu()
                    labels_pred = output["labels"].detach().cpu()
                    
                    # Filter by confidence threshold
                    keep = scores >= CONFIDENCE_THRESHOLD
                    boxes_filtered = boxes[keep]
                    scores_filtered = scores[keep]
                    labels_filtered = labels_pred[keep]
                    
                    # Convert labels back
                    if len(labels_filtered) > 0:
                        labels_filtered = labels_filtered - 1
                    
                    # Ground truth
                    gt_boxes = bboxs[img_idx]
                    gt_labels = labels[img_idx]
                    
                    # Statistics
                    gt_moving = (gt_labels == 1).sum().item()
                    gt_non_moving = (gt_labels == 0).sum().item()
                    pred_moving = (labels_filtered == 1).sum().item() if len(labels_filtered) > 0 else 0
                    pred_non_moving = (labels_filtered == 0).sum().item() if len(labels_filtered) > 0 else 0
                    
                    print(f"\n{image_type} Image (Batch {batch_idx}, Img {img_idx}):")
                    print(f"  GT: {gt_non_moving} non-moving, {gt_moving} moving")
                    print(f"  Pred: {pred_non_moving} non-moving, {pred_moving} moving")
                    if len(scores_filtered) > 0:
                        print(f"  Avg confidence: {scores_filtered.mean():.4f}")
                    
                    # Visualize
                    image_np = images[img_idx].permute(1, 2, 0).numpy()
                    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Ground Truth (LEFT)
                    axes[0].imshow(image_np)
                    axes[0].set_title(f"Ground Truth - {image_type}\n{gt_non_moving} non-moving, {gt_moving} moving", 
                                     fontsize=12, fontweight='bold')
                    for box, label in zip(gt_boxes, gt_labels):
                        xtl, ytl, xbr, ybr = box
                        color = COLOR_GT_NON_MOVING if label.item() == 0 else COLOR_GT_MOVING
                        axes[0].add_patch(
                            plt.Rectangle((xtl, ytl), xbr - xtl, ybr - ytl,
                                        edgecolor=color, facecolor='none', linewidth=2)
                        )
                    axes[0].axis('off')
                    
                    # Predictions (RIGHT)
                    axes[1].imshow(image_np)
                    axes[1].set_title(f"Model Predictions - {image_type} (score>={CONFIDENCE_THRESHOLD})\n{pred_non_moving} non-moving, {pred_moving} moving", 
                                     fontsize=12, fontweight='bold')
                    for box, score, label in zip(boxes_filtered, scores_filtered, labels_filtered):
                        xtl, ytl, xbr, ybr = box
                        color = COLOR_PRED_NON_MOVING if label.item() == 0 else COLOR_PRED_MOVING
                        axes[1].add_patch(
                            plt.Rectangle((xtl, ytl), xbr - xtl, ybr - ytl,
                                        edgecolor=color, facecolor='none', linewidth=2)
                        )
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    break

    def save_model(self, filename="fasterrcnn_moving_objects.pth"):
        """Save model weights."""
        path = f"{MODEL_DIR}/{filename}"
        torch.save(self.model.state_dict(), path)
        print(f"\nModel saved to: {path}")


def main():
    """Main training and evaluation pipeline."""
    # Initialize trainer
    trainer = BaseballModelTrainer()
    
    # Precategorize test set
    moving_indices, non_moving_indices = trainer._precategorize_test_set()
    
    # Train model
    trainer.train()
    
    # Evaluate model
    trainer.evaluate()
    
    # Visualize results
    trainer.visualize_samples(moving_indices, non_moving_indices)
    
    # Save final model
    trainer.save_model()


if __name__ == "__main__":
    main()
