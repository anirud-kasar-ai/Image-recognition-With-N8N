"""
Face Augmentation API - OpenCV Only (No Dlib Required)
Enhanced face detection with multiple cascades for better accuracy
Install: pip install flask opencv-python pillow numpy
"""

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import base64
import numpy as np
import math
import cv2

app = Flask(__name__)

print("🔍 Initializing OpenCV face detection cascades...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
print("✅ OpenCV ready!")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def base64_to_image(base64_str):
    """Convert base64 string to PIL Image"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data)).convert('RGB')

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode()

def get_facial_landmarks(image):
    """Enhanced OpenCV face detection with better landmark estimation"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with multiple scale factors for better results
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            # Try with different parameters
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract face region for eye detection
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = img_array[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        
        points = {}
        face_center_x = x + w // 2
        
        # Eye detection and positioning
        if len(eyes) >= 2:
            # Sort eyes left to right
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[1] if len(eyes) > 1 else eyes[0]
            
            points['left_eye_center'] = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
            points['right_eye_center'] = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
        else:
            # Estimated eye positions based on face geometry
            eye_y = y + int(h * 0.37)
            points['left_eye_center'] = (x + int(w * 0.32), eye_y)
            points['right_eye_center'] = (x + int(w * 0.68), eye_y)
        
        # Calculate eye distance for proportional sizing
        eye_distance = math.dist(points['left_eye_center'], points['right_eye_center'])
        
        # Enhanced landmark estimation based on facial proportions
        # Eyebrows (slightly above eyes)
        eyebrow_offset = int(eye_distance * 0.25)
        points['left_eyebrow'] = (points['left_eye_center'][0], points['left_eye_center'][1] - eyebrow_offset)
        points['right_eyebrow'] = (points['right_eye_center'][0], points['right_eye_center'][1] - eyebrow_offset)
        
        # Nose positions (using golden ratio proportions)
        nose_y_start = y + int(h * 0.42)
        nose_y_tip = y + int(h * 0.58)
        points['nose_bridge'] = (face_center_x, nose_y_start)
        points['nose_tip'] = (face_center_x, nose_y_tip)
        points['nose_left'] = (face_center_x - int(eye_distance * 0.15), nose_y_tip)
        points['nose_right'] = (face_center_x + int(eye_distance * 0.15), nose_y_tip)
        
        # Mouth positions (proportional to face)
        mouth_y = y + int(h * 0.70)
        mouth_width = int(eye_distance * 0.8)
        points['mouth_center'] = (face_center_x, mouth_y)
        points['mouth_left'] = (face_center_x - mouth_width//2, mouth_y)
        points['mouth_right'] = (face_center_x + mouth_width//2, mouth_y)
        points['mouth_top'] = (face_center_x, y + int(h * 0.66))
        points['mouth_bottom'] = (face_center_x, y + int(h * 0.74))
        
        # Jaw and chin
        points['chin'] = (face_center_x, y + int(h * 0.96))
        points['jaw_left'] = (x + int(w * 0.12), y + int(h * 0.78))
        points['jaw_right'] = (x + int(w * 0.88), y + int(h * 0.78))
        
        # Forehead
        points['forehead'] = (face_center_x, y + int(h * 0.12))
        
        # Store face dimensions for scaling
        points['_face_width'] = w
        points['_face_height'] = h
        points['_eye_distance'] = eye_distance
        
        return points
        
    except Exception as e:
        print(f"⚠️  Face detection error: {e}")
        return None

def visualize_landmarks(image, landmarks):
    """Debug: Draw landmarks on image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Filter out internal metadata
    visible_landmarks = {k: v for k, v in landmarks.items() if not k.startswith('_')}
    
    for name, (x, y) in visible_landmarks.items():
        # Draw point
        draw.ellipse([x-4, y-4, x+4, y+4], fill='red', outline='yellow', width=2)
        # Draw label
        draw.text((x+6, y-8), name.replace('_', ' '), fill='white')
    
    return img_copy

# ============================================================================
# AUGMENTATION FUNCTIONS WITH PRECISE FACE MAPPING
# ============================================================================

def add_sunglasses(image, landmarks):
    """Add sunglasses with precise eye alignment"""
    img_rgba = image.convert('RGBA')
    scale = 2
    overlay = Image.new('RGBA', (img_rgba.size[0] * scale, img_rgba.size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    # Get eye positions
    left_eye = (landmarks['left_eye_center'][0] * scale, landmarks['left_eye_center'][1] * scale)
    right_eye = (landmarks['right_eye_center'][0] * scale, landmarks['right_eye_center'][1] * scale)
    
    # Size based on eye distance
    eye_dist = landmarks['_eye_distance'] * scale
    lens_w = int(eye_dist * 0.50)
    lens_h = int(lens_w * 0.76)
    
    # Draw both lenses
    for eye_pos in [left_eye, right_eye]:
        # Gradient dark lenses
        for i in range(lens_h):
            progress = i / lens_h
            alpha = 250 if 0.10 <= progress <= 0.90 else int(200 + (progress if progress < 0.10 else (1-progress)) * 180)
            draw.ellipse(
                [eye_pos[0] - lens_w//2, eye_pos[1] - lens_h//2 + i,
                 eye_pos[0] + lens_w//2, eye_pos[1] - lens_h//2 + i + 2],
                fill=(2, 2, 2, alpha)
            )
        
        # Frame around each lens
        for thickness in range(8, 0, -1):
            alpha_frame = int(255 - (8 - thickness) * 18)
            draw.ellipse(
                [eye_pos[0] - lens_w//2 - thickness, eye_pos[1] - lens_h//2 - thickness,
                 eye_pos[0] + lens_w//2 + thickness, eye_pos[1] + lens_h//2 + thickness],
                outline=(0, 0, 0, alpha_frame), width=2
            )
        
        # Reflective highlight
        hl_x, hl_y = eye_pos[0] - lens_w//4, eye_pos[1] - lens_h//3
        for size in range(18, 10, -2):
            draw.ellipse([hl_x - size, hl_y - size, hl_x + size, hl_y + size],
                        fill=(220, 220, 220, int(200 - (18 - size) * 14)))
    
    # Bridge connecting lenses
    bridge_left = left_eye[0] + lens_w//2
    bridge_right = right_eye[0] - lens_w//2
    bridge_y = (left_eye[1] + right_eye[1]) // 2
    bridge_h = int(lens_h * 0.26)
    
    for i in range(max(1, bridge_right - bridge_left)):
        draw.rectangle([bridge_left + i, bridge_y - bridge_h//2,
                       bridge_left + i + 1, bridge_y + bridge_h//2],
                      fill=(5, 5, 5, 255))
    
    # Temple arms extending to sides
    for i in range(12):
        alpha = int(255 - i * 18)
        # Left arm
        draw.line([left_eye[0] - lens_w//2 - 6, bridge_y + i - 6, 5, bridge_y + i - 6],
                 fill=(0, 0, 0, alpha), width=4)
        # Right arm
        draw.line([right_eye[0] + lens_w//2 + 6, bridge_y + i - 6,
                  overlay.size[0] - 5, bridge_y + i - 6],
                 fill=(0, 0, 0, alpha), width=4)
    
    overlay = overlay.resize(img_rgba.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=0.4))
    result = Image.alpha_composite(img_rgba, overlay)
    return result.convert('RGB')

def add_mask(image, landmarks):
    """Add surgical mask with precise face fitting"""
    img_rgba = image.convert('RGBA')
    scale = 2
    overlay = Image.new('RGBA', (img_rgba.size[0] * scale, img_rgba.size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    # Scale all landmarks
    nose_bridge = tuple(v * scale for v in landmarks['nose_bridge'])
    nose_tip = tuple(v * scale for v in landmarks['nose_tip'])
    chin = tuple(v * scale for v in landmarks['chin'])
    jaw_left = tuple(v * scale for v in landmarks['jaw_left'])
    jaw_right = tuple(v * scale for v in landmarks['jaw_right'])
    mouth_bottom = tuple(v * scale for v in landmarks['mouth_bottom'])
    
    face_width = landmarks['_face_width'] * scale
    
    # Precise mask positioning
    mask_top = int(nose_bridge[1] + (nose_tip[1] - nose_bridge[1]) * 0.12)
    mask_bottom = int(mouth_bottom[1] + (chin[1] - mouth_bottom[1]) * 0.42)
    
    # Face-fitted mask shape
    mask_points = [
        (jaw_left[0] + int(face_width * 0.14), mask_top + 6),
        (nose_tip[0] - int(face_width * 0.14), mask_top - 10),
        (nose_tip[0], mask_top - 20),
        (nose_tip[0] + int(face_width * 0.14), mask_top - 10),
        (jaw_right[0] - int(face_width * 0.14), mask_top + 6),
        (jaw_right[0] - int(face_width * 0.08), mask_bottom - 10),
        (chin[0] + int(face_width * 0.12), mask_bottom + 2),
        (chin[0], mask_bottom + 6),
        (chin[0] - int(face_width * 0.12), mask_bottom + 2),
        (jaw_left[0] + int(face_width * 0.08), mask_bottom - 10)
    ]
    
    # Base mask with subtle layering
    for y_off in range(-2, 3):
        pts = [(p[0], p[1] + y_off) for p in mask_points]
        draw.polygon(pts, fill=(244 + y_off, 249, 255, int(253 + y_off * 2)))
    
    # Realistic pleats
    mask_h = mask_bottom - mask_top
    for i in range(4):
        pleat_y = mask_top + int(mask_h * (0.22 + i * 0.20))
        pts = [(x, pleat_y + int(4 * math.sin((x - jaw_left[0]) / max(1, jaw_right[0] - jaw_left[0]) * math.pi)))
               for x in range(int(jaw_left[0] + face_width * 0.14), int(jaw_right[0] - face_width * 0.14), 8)]
        if len(pts) > 1:
            draw.line(pts, fill=(220, 230, 240, 210), width=3)
            draw.line([(p[0], p[1] + 2) for p in pts], fill=(208, 218, 228, 150), width=2)
    
    # Border outline
    for t in range(4, 0, -1):
        draw.polygon(mask_points, outline=(200, 210, 222, int(255 - (4 - t) * 30)), width=t)
    
    # Ear loops
    loop_y = mask_top + int(mask_h * 0.44)
    for side_x, angles in [(jaw_left[0], (90, 270)), (jaw_right[0], (270, 450))]:
        for t in range(5, 0, -1):
            loop_offset = int(face_width * 0.16)
            bbox = ([side_x - loop_offset - t, loop_y - 14, side_x + int(face_width * 0.09) - t, loop_y + 14] 
                   if side_x == jaw_left[0]
                   else [side_x - int(face_width * 0.09) + t, loop_y - 14, side_x + loop_offset + t, loop_y + 14])
            draw.arc(bbox, angles[0], angles[1], fill=(234, 241, 251, int(248 - (5 - t) * 16)), width=2)
    
    # Nose wire detail
    wire_width = int(face_width * 0.12)
    for i in range(6):
        draw.rectangle([nose_tip[0] - wire_width, mask_top + 2 + i, 
                       nose_tip[0] + wire_width, mask_top + 6 + i],
                      fill=(208 + i * 3, 208 + i * 3, 213 + i * 3, int(242 - i * 12)))
    
    overlay = overlay.resize(img_rgba.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=0.6))
    result = Image.alpha_composite(img_rgba, overlay)
    return result.convert('RGB')

def add_beard(image, landmarks):
    """Add realistic beard with precise face fitting"""
    img_rgba = image.convert('RGBA')
    scale = 2
    overlay = Image.new('RGBA', (img_rgba.size[0] * scale, img_rgba.size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    # Scale landmarks
    mouth_bottom = tuple(v * scale for v in landmarks['mouth_bottom'])
    chin = tuple(v * scale for v in landmarks['chin'])
    jaw_left = tuple(v * scale for v in landmarks['jaw_left'])
    jaw_right = tuple(v * scale for v in landmarks['jaw_right'])
    
    face_width = landmarks['_face_width'] * scale
    
    # Precise beard positioning
    beard_start_y = mouth_bottom[1] - 2  # Start just below mouth
    beard_end_y = int(chin[1] + (chin[1] - mouth_bottom[1]) * 0.48)
    
    # Natural beard shape proportional to face
    beard_pts = [
        (jaw_left[0] + int(face_width * 0.16), beard_start_y),
        (jaw_left[0] + int(face_width * 0.11), beard_start_y + int((beard_end_y - beard_start_y) * 0.08)),
        (jaw_left[0] + int(face_width * 0.07), chin[1] - 5),
        (jaw_left[0] + int(face_width * 0.06), beard_end_y - 5),
        (chin[0] - int(face_width * 0.13), beard_end_y + 14),
        (chin[0], beard_end_y + 22),
        (chin[0] + int(face_width * 0.13), beard_end_y + 14),
        (jaw_right[0] - int(face_width * 0.06), beard_end_y - 5),
        (jaw_right[0] - int(face_width * 0.07), chin[1] - 5),
        (jaw_right[0] - int(face_width * 0.11), beard_start_y + int((beard_end_y - beard_start_y) * 0.08)),
        (jaw_right[0] - int(face_width * 0.16), beard_start_y)
    ]
    
    # Beard color
    base_c = (44, 34, 24)
    
    # Base layers for depth
    for layer in range(3):
        pts = [(p[0] + layer * 2, p[1] + layer * 2) for p in beard_pts]
        draw.polygon(pts, fill=(*base_c, int(230 - layer * 9)))
    
    # Realistic hair strands - scale with face size
    np.random.seed(42)
    num_strands = int(480 * (face_width / 400))
    
    for _ in range(num_strands):
        x = int(jaw_left[0] + face_width * 0.09 + np.random.random() * face_width * 0.82)
        y = int(beard_start_y + np.random.random() * (beard_end_y - beard_start_y + 22))
        length = np.random.randint(11, 30)
        angle = np.random.randint(-38, 38)
        
        # Color variation
        color_var = np.random.randint(-14, 40)
        hair_c = (min(255, max(0, base_c[0] + color_var)),
                 min(255, max(0, base_c[1] + color_var)),
                 min(255, max(0, base_c[2] + color_var - 2)),
                 np.random.randint(195, 254))
        
        end_x = x + int(length * 0.30 * np.sin(np.radians(angle)))
        end_y = y + int(length * np.cos(np.radians(angle)))
        draw.line([(x, y), (end_x, end_y)], fill=hair_c, width=np.random.choice([1, 1, 1, 2]))
    
    # Highlight strands for depth
    for _ in range(int(75 * (face_width / 400))):
        x = int(jaw_left[0] + face_width * 0.12 + np.random.random() * face_width * 0.76)
        y = int(beard_start_y + np.random.random() * (beard_end_y - beard_start_y))
        length = np.random.randint(9, 17)
        draw.line([(x, y), (x + np.random.randint(-3, 3), y + length)],
                 fill=(90, 80, 70, 205), width=1)
    
    overlay = overlay.resize(img_rgba.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=0.42))
    result = Image.alpha_composite(img_rgba, overlay)
    return result.convert('RGB')

def add_tilt_left(image, landmarks):
    """Tilt image left with dramatic lighting"""
    img_rgba = image.convert('RGBA')
    rotated = img_rgba.rotate(15, expand=False, fillcolor=(255, 255, 255, 0))
    
    w, h = rotated.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    for x in range(w):
        progress = x / w
        if progress > 0.5:
            brightness = int((progress - 0.5) * 2 * 100)
            draw.line([(x, 0), (x, h)], fill=(255, 250, 200, brightness), width=1)
        else:
            darkness = int((0.5 - progress) * 2 * 65)
            draw.line([(x, 0), (x, h)], fill=(30, 50, 80, darkness), width=1)
    
    result = Image.alpha_composite(rotated, overlay)
    enhancer = ImageEnhance.Contrast(result.convert('RGB'))
    return enhancer.enhance(1.10)

def add_tilt_right(image, landmarks):
    """Tilt image right with dramatic lighting"""
    img_rgba = image.convert('RGBA')
    rotated = img_rgba.rotate(-15, expand=False, fillcolor=(255, 255, 255, 0))
    
    w, h = rotated.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    for x in range(w):
        progress = x / w
        if progress < 0.5:
            brightness = int((0.5 - progress) * 2 * 100)
            draw.line([(x, 0), (x, h)], fill=(255, 250, 200, brightness), width=1)
        else:
            darkness = int((progress - 0.5) * 2 * 65)
            draw.line([(x, 0), (x, h)], fill=(30, 50, 80, darkness), width=1)
    
    result = Image.alpha_composite(rotated, overlay)
    enhancer = ImageEnhance.Contrast(result.convert('RGB'))
    return enhancer.enhance(1.10)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/augment', methods=['POST'])
def augment():
    """
    Face Augmentation with OpenCV Face Mapping
    
    Input:
    {
        "img_base64": "base64_string",
        "fileName": "photo.jpg",
        "employeeName": "John Doe",
        "augmentations": ["sunglasses", "mask", "beard"] or ["all"],
        "debug": false  // Set true to see landmark visualization
    }
    """
    try:
        data = request.json
        
        img_base64 = data.get('img_base64')
        file_name = data.get('fileName', 'image.jpg')
        employee_name = data.get('employeeName', 'Unknown')
        augmentation_types = data.get('augmentations', ['all'])
        debug = data.get('debug', False)
        
        if not img_base64:
            return jsonify({'error': 'img_base64 is required'}), 400
        
        print(f"\n{'='*60}")
        print(f"📸 Processing: {file_name}")
        print(f"👤 Employee: {employee_name}")
        print(f"{'='*60}")
        
        original = base64_to_image(img_base64)
        
        # Detect facial landmarks
        print("🔍 Detecting face and landmarks...", end=' ')
        landmarks = get_facial_landmarks(original)
        
        if not landmarks:
            print("❌ No face detected!")
            return jsonify({'error': 'No face detected in image. Please ensure the image contains a clear frontal face.'}), 400
        
        visible_count = len([k for k in landmarks.keys() if not k.startswith('_')])
        print(f"✅ Face found with {visible_count} landmarks")
        
        # Debug mode: return image with landmarks
        if debug:
            debug_img = visualize_landmarks(original, landmarks)
            return jsonify([{
                'fileName': f"{file_name.rsplit('.', 1)[0]}_landmarks.jpg",
                'employeeName': employee_name,
                'img_base64': image_to_base64(debug_img),
                'landmarks': {k: list(v) if isinstance(v, tuple) else v for k, v in landmarks.items()}
            }])
        
        # Define augmentations
        augmentations = {
            'sunglasses': add_sunglasses,
            'mask': add_mask,
            'beard': add_beard,
            'tilt_left': add_tilt_left,
            'tilt_right': add_tilt_right
        }
        
        if 'all' in augmentation_types:
            augmentation_types = ['original'] + list(augmentations.keys())
        
        results = []
        
        for aug_type in augmentation_types:
            if aug_type == 'original':
                results.append({
                    'fileName': file_name,
                    'employeeName': employee_name,
                    'img_base64': img_base64
                })
                print(f"✅ original → {file_name}")
                continue
            
            if aug_type in augmentations:
                try:
                    print(f"🎨 Processing {aug_type}...", end=' ')
                    augmented_image = augmentations[aug_type](original.copy(), landmarks)
                    
                    base_name, ext = file_name.rsplit('.', 1) if '.' in file_name else (file_name, 'jpg')
                    new_filename = f"{base_name}_{aug_type}.{ext}"
                    
                    results.append({
                        'fileName': new_filename,
                        'employeeName': employee_name,
                        'img_base64': image_to_base64(augmented_image)
                    })
                    print(f"✅ → {new_filename}")
                    
                except Exception as e:
                    print(f"❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'fileName': f"{file_name}_{aug_type}_FAILED",
                        'employeeName': employee_name,
                        'img_base64': img_base64,
                        'error': str(e)
                    })
        
        print(f"\n✅ Complete: {len(results)} images generated")
        print(f"{'='*60}\n")
        
        return jsonify(results)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': 'opencv-only-v1',
        'detection': 'opencv-enhanced',
        'outputs': ['fileName', 'employeeName', 'img_base64'],
        'augmentations': ['sunglasses', 'mask', 'beard', 'tilt_left', 'tilt_right'],
        'debug_mode': 'available'
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 FACE AUGMENTATION API - OpenCV Enhanced")
    print("="*70)
    print("🔍 Detection: OpenCV Haar Cascade (No dlib required)")
    print("📍 Smart face mapping for all angles and sizes")
    print("📤 Output: [{ fileName, employeeName, img_base64 }, ...]")
    print("🐛 Debug mode: Set 'debug': true to visualize landmarks")
    print("🌐 http://localhost:3001")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=3001, debug=True)