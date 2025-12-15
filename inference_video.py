import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# Import configuration
from config import (
    MODEL_NUM_CLASSES, MODEL_DIR, TRAINED_MODEL_PATH, 
    CONFIDENCE_THRESHOLD, COLOR_PRED_MOVING, COLOR_PRED_NON_MOVING
)

# Define input and output folders
INPUT_VIDEO_FOLDER = 'new_videos'
OUTPUT_VIDEO_FOLDER = 'output_videos'

def get_model(device):
    """Initialize and load the trained model."""
    print("Initializing model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        num_classes=91
    )
    
    # Replace classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=MODEL_NUM_CLASSES)
    
    # Load trained weights
    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"Loading weights from {TRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    else:
        print(f"Error: Model file not found at {TRAINED_MODEL_PATH}")
        return None

    model.to(device)
    model.eval()
    return model

def process_video(video_path, output_path, model, device):
    """Process a single video frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {os.path.basename(video_path)} ({total_frames} frames)...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        
        # Transform to tensor
        image_tensor = torchvision.transforms.functional.to_tensor(image_pil).to(device)
        
        # Inference
        with torch.no_grad():
            prediction = model([image_tensor])[0]

        # Filter predictions
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()

        # Draw detections on the original frame (BGR)
        for box, score, label in zip(boxes, scores, labels):
            if score >= CONFIDENCE_THRESHOLD:
                xtl, ytl, xbr, ybr = box.astype(int)
                
                # Determine color and label text
                if label == 1: # Non-moving
                    color = (0, 255, 0) # Green (BGR)
                    label_text = f"Non-Moving: {score:.2f}"
                elif label == 2: # Moving
                    color = (128, 0, 128) # Purple (BGR)
                    label_text = f"Moving: {score:.2f}"
                else:
                    color = (255, 255, 255)
                    label_text = f"Class {label}: {score:.2f}"

                # Draw rectangle
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, 2)
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (xtl, ytl - 20), (xtl + w, ytl), color, -1)
                
                # Draw text
                cv2.putText(frame, label_text, (xtl, ytl - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Saved processed video to {output_path}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    model = get_model(device)
    if model is None:
        return

    # Create output directory
    os.makedirs(OUTPUT_VIDEO_FOLDER, exist_ok=True)

    # Process all videos in input folder
    if not os.path.exists(INPUT_VIDEO_FOLDER):
        print(f"Input folder '{INPUT_VIDEO_FOLDER}' does not exist.")
        return

    video_files = [f for f in os.listdir(INPUT_VIDEO_FOLDER) if f.lower().endswith(('.mov'))]
    
    if not video_files:
        print(f"No video files found in '{INPUT_VIDEO_FOLDER}'")
        return

    for video_file in video_files:
        input_path = os.path.join(INPUT_VIDEO_FOLDER, video_file)
        output_path = os.path.join(OUTPUT_VIDEO_FOLDER, f"processed_{video_file}")
        
        # Change extension to mp4 for output if needed
        output_path = os.path.splitext(output_path)[0] + ".mp4"
        
        process_video(input_path, output_path, model, device)

    print("\nAll videos processed!")

if __name__ == "__main__":
    main()
