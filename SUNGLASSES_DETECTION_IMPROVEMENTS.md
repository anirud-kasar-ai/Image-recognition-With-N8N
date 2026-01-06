# Sunglasses Detection Improvements

## Problem Summary
- Sunglasses detection was failing to reliably detect when people wore sunglasses
- This caused the n8n workflow to route to the "without sunglasses" path incorrectly
- When routed wrong, embedding similarity scores were low and face recognition failed

## Root Causes Identified
1. **Low confidence thresholds** - The detection wasn't aggressive enough
2. **Weak detection methods** - Limited to basic darkness detection
3. **Poor weight distribution** - Important features weren't prioritized
4. **Missing key metrics** - Didn't capture eye region symmetry and structure

## Solutions Implemented

### 1. **Enhanced Detection Algorithm** (8 Methods)
- **Darkness Detection (30% weight)** - Sunglasses block light significantly
- **Contrast Detection (20% weight)** - Sunglasses create uniform, low-contrast areas
- **Histogram Analysis (15% weight)** - Flat histograms indicate uniform darkness
- **Edge Detection (12% weight)** - Frames create moderate edge patterns
- **Color Saturation (12% weight)** - Reduced saturation in dark areas
- **Cascade Classifier (8% weight)** - Haar cascade for eyeglasses detection
- **Symmetry Analysis (3% weight)** - Both eyes equally dark
- **Specular Highlights** - Some sunglasses show light reflections

### 2. **Improved Image Processing**
- Added CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
- Expanded eye detection region (5%-65% of face height)
- More precise edge detection thresholds (Canny: 20-100)

### 3. **Better Threshold Tuning**
```python
# Before:
- min_confidence: 0.60 (too high)
- dark_threshold: 70 (too lenient)

# After:
- min_confidence: 0.45 (more sensitive)
- dark_threshold: 85 (more aggressive)
```

### 4. **Hard Rules for Real Sunglasses**
```python
if mean_intensity < 65:
    score = max(score, 0.75)  # Very dark = likely sunglasses
elif mean_intensity < 85:
    score = max(score, 0.60)  # Dark = probable sunglasses
```

### 5. **Cascade Boosting**
- If Haar cascade detects eyeglasses, multiply confidence by 1.3
- Even single cascade hit significantly boosts final score

### 6. **Symmetry Detection**
- Compare left vs right eye intensity
- Both equally dark = strong sunglasses indicator

## Key Metrics Returned
The detection now returns detailed metrics for debugging:
- `mean_intensity` - Average brightness of eye region
- `darkness_score`, `contrast_score`, `histogram_score` - Individual method scores
- `left_eye_intensity`, `right_eye_intensity` - Asymmetry detection
- `cascade_hits` - Number of Haar cascade detections
- `final_components` - All component scores for n8n workflow analysis

## Expected Improvements
✅ Better detection of sunglasses wearers  
✅ Higher confidence scores for true positives  
✅ Fewer false negatives (missing sunglasses)  
✅ Correct routing in n8n workflows  
✅ Better embedding quality for sunglasses cases  
✅ Improved face recognition accuracy  

## Testing
Run the test script to validate improvements:
```bash
python test_sunglass_detection.py          # Test all images
python test_sunglass_detection.py <image>  # Test specific image
```

## N8N Workflow Integration
The API endpoint `/process_video` now returns:
```json
{
  "appearance": {
    "sunglasses": true/false,
    "sunglasses_confidence": 0.0-1.0,
    "sunglasses_details": {...}
  }
}
```

Use `sunglasses_confidence` in your n8n IF node:
- **Threshold = 0.45** for sunglasses workflow (more lenient)
- **Threshold > 0.50** for without sunglasses workflow

## Debug Endpoint
New endpoint for testing: `POST /test_sunglasses`
```json
{
  "image_base64": "..."
}
```
Returns detailed detection metrics for each face.

## Configuration
Adjust these in `multi_face.py` if needed:
```python
SUNGLASSES_ADVANCED = {
    "min_confidence": 0.45,      # Lower = more sensitive
    "dark_threshold": 85,        # Lower = darker detection
    # ... other params
}
```

## Next Steps
1. Test with your actual sunglasses images
2. Adjust thresholds if needed based on results
3. Monitor embedding quality improvements
4. Tune n8n workflow thresholds if necessary
