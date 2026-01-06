#!/usr/bin/env python3
"""
Test script for improved sunglasses detection.
Tests the detection on images with and without sunglasses.
"""
import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the detection function
from multi_face import detect_sunglasses_advanced, crop_face_with_padding, SUNGLASSES_ADVANCED
from ultralytics import YOLO

def test_sunglasses_on_image(image_path):
    """Test sunglasses detection on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return False
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return False
    
    print(f"Image shape: {img.shape}")
    
    # Load YOLO face detector
    try:
        detector = YOLO('yolov8n-face.pt')
    except Exception as e:
        print(f"ERROR: Could not load YOLO model: {e}")
        return False
    
    # Detect faces
    results = detector(img, conf=0.55)
    
    if not results or len(results[0].boxes) == 0:
        print("No faces detected")
        return False
    
    num_faces = len(results[0].boxes)
    print(f"\nDetected {num_faces} face(s)")
    
    success = True
    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        print(f"\n--- Face {idx + 1} ---")
        print(f"YOLO Confidence: {conf:.4f}")
        print(f"BBox: ({x1}, {y1}) -> ({x2}, {y2})")
        
        # Crop face
        crop = crop_face_with_padding(img, (x1, y1, x2, y2))
        
        # Test sunglasses detection with debug enabled
        result = detect_sunglasses_advanced(crop, debug=True)
        
        print(f"\nSunglasses Detection Results:")
        print(f"  Detected: {result['sunglasses']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Min threshold: {SUNGLASSES_ADVANCED['min_confidence']}")
        
        # Print key metrics
        details = result['details']
        print(f"\nKey Metrics:")
        print(f"  Mean Intensity: {details['mean_intensity']:.2f}")
        print(f"  Darkness Score: {details['darkness_score']:.3f}")
        print(f"  Contrast Score: {details['contrast_score']:.3f}")
        print(f"  Histogram Score: {details['histogram_score']:.3f}")
        print(f"  Edge Score: {details['edge_score']:.3f}")
        print(f"  Color Score: {details['color_score']:.3f}")
        print(f"  Symmetry Score: {details['symmetry_score']:.3f}")
        print(f"  Cascade Hits: {details['cascade_hits']}")
        print(f"  Left Eye Intensity: {details['left_eye_intensity']:.2f}")
        print(f"  Right Eye Intensity: {details['right_eye_intensity']:.2f}")
        
    return success


def test_all_images():
    """Test detection on all images in the images/ folder."""
    image_dir = Path("images")
    
    if not image_dir.exists():
        print(f"ERROR: images/ directory not found")
        return False
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    results = {}
    for img_path in sorted(image_files):
        try:
            test_sunglasses_on_image(str(img_path))
            results[str(img_path)] = "OK"
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            results[str(img_path)] = f"ERROR: {e}"
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for path, status in results.items():
        print(f"{path}: {status}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUNGLASSES DETECTION TEST SUITE")
    print("="*60)
    print(f"Config - Min Confidence: {SUNGLASSES_ADVANCED['min_confidence']}")
    print(f"Config - Dark Threshold: {SUNGLASSES_ADVANCED['dark_threshold']}")
    
    if len(sys.argv) > 1:
        # Test specific image
        test_sunglasses_on_image(sys.argv[1])
    else:
        # Test all images
        test_all_images()
