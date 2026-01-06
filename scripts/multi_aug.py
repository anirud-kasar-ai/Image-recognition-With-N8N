"""
FREE Face Augmentation - NO DOWNLOADS NEEDED!
Uses OpenCV's built-in Haar Cascades
Install: pip install opencv-python pillow numpy
"""

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
import numpy as np
import math
import cv2

app = Flask(__name__)

def base64_to_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data)).convert('RGB')

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode()

def get_facial_landmarks_opencv(image):
    """Get facial features using OpenCV Haar Cascades (built-in, no download!)"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Load built-in Haar Cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("⚠️ No face detected")
            return None
        
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Detect eyes within face region
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10)
        
        # Sort eyes left to right
        eyes = sorted(eyes, key=lambda e: e[0])
        
        points = {}
        
        if len(eyes) >= 2:
            # Use detected eyes
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            points['left_eye_center'] = (
                x + left_eye[0] + left_eye[2]//2,
                y + left_eye[1] + left_eye[3]//2
            )
            points['right_eye_center'] = (
                x + right_eye[0] + right_eye[2]//2,
                y + right_eye[1] + right_eye[3]//2
            )
        else:
            # Estimate eye positions based on face
            eye_y = y + int(h * 0.35)
            points['left_eye_center'] = (x + int(w * 0.30), eye_y)
            points['right_eye_center'] = (x + int(w * 0.70), eye_y)
        
        # Estimate eyebrow positions (above eyes)
        eyebrow_offset = int(h * 0.08)  # Distance from eye to eyebrow
        points['left_eyebrow'] = (points['left_eye_center'][0], points['left_eye_center'][1] - eyebrow_offset)
        points['right_eyebrow'] = (points['right_eye_center'][0], points['right_eye_center'][1] - eyebrow_offset)
        
        # Estimate other facial features based on face rectangle
        face_center_x = x + w // 2
        
        # Nose (lower middle of face)
        points['nose_bridge'] = (face_center_x, y + int(h * 0.40))
        points['nose_tip'] = (face_center_x, y + int(h * 0.55))
        
        # Mouth
        mouth_y = y + int(h * 0.70)
        points['mouth_left'] = (x + int(w * 0.35), mouth_y)
        points['mouth_right'] = (x + int(w * 0.65), mouth_y)
        points['mouth_top'] = (face_center_x, y + int(h * 0.65))
        points['mouth_bottom'] = (face_center_x, y + int(h * 0.75))
        
        # Chin and jaw
        points['chin'] = (face_center_x, y + int(h * 0.95))
        points['jaw_left'] = (x + int(w * 0.15), y + int(h * 0.75))
        points['jaw_right'] = (x + int(w * 0.85), y + int(h * 0.75))
        
        # Forehead
        points['forehead'] = (face_center_x, y + int(h * 0.15))
        
        print(f"✅ OpenCV detected face at ({x},{y}) size {w}x{h}")
        print(f"   Left eye: {points['left_eye_center']}")
        print(f"   Right eye: {points['right_eye_center']}")
        print(f"   Nose: {points['nose_tip']}")
        
        return points
        
    except Exception as e:
        print(f"❌ OpenCV detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_premium_sunglasses(image):
    """Sunglasses positioned exactly on eyes"""
    landmarks = get_facial_landmarks_opencv(image)
    
    if not landmarks:
        print("   Using fallback positioning")
        w, h = image.size
        landmarks = {
            'left_eye_center': (int(w * 0.35), int(h * 0.40)),
            'right_eye_center': (int(w * 0.65), int(h * 0.40))
        }
    else:
        print("   ✓ Using OpenCV landmarks for sunglasses")
    
    img_rgba = image.convert('RGBA')
    scale = 2
    overlay = Image.new('RGBA', (img_rgba.size[0] * scale, img_rgba.size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    left_eye = (landmarks['left_eye_center'][0] * scale, landmarks['left_eye_center'][1] * scale)
    right_eye = (landmarks['right_eye_center'][0] * scale, landmarks['right_eye_center'][1] * scale)
    
    eye_dist = math.dist(left_eye, right_eye)
    lens_w = int(eye_dist * 0.45)  # Increased from 0.35 to 0.45 (30% bigger)
    lens_h = int(lens_w * 0.85)  # Increased height ratio to cover more area
    
    print(f"   Drawing at: L{left_eye}, R{right_eye}, lens={lens_w}x{lens_h}")
    
    # Draw lenses
    for eye_pos in [left_eye, right_eye]:
        # Dark gradient lens
        for i in range(lens_h):
            progress = i / lens_h
            if progress < 0.15:
                alpha = int(180 + progress * 200)
            elif progress > 0.85:
                alpha = int(240 - (progress - 0.85) * 400)
            else:
                alpha = 245
            
            draw.ellipse(
                [eye_pos[0] - lens_w//2, eye_pos[1] - lens_h//2 + i,
                 eye_pos[0] + lens_w//2, eye_pos[1] - lens_h//2 + i + 3],
                fill=(2, 2, 2, alpha)
            )
        
        # Frame
        for thickness in range(8, 0, -1):
            alpha_frame = int(255 - (8 - thickness) * 20)
            draw.ellipse(
                [eye_pos[0] - lens_w//2 - thickness, eye_pos[1] - lens_h//2 - thickness,
                 eye_pos[0] + lens_w//2 + thickness, eye_pos[1] + lens_h//2 + thickness],
                outline=(0, 0, 0, alpha_frame), width=2
            )
        
        # Reflection highlight
        hl_x, hl_y = eye_pos[0] - lens_w//4, eye_pos[1] - lens_h//4
        for size in range(16, 8, -2):
            draw.ellipse([hl_x - size, hl_y - size, hl_x + size, hl_y + size],
                        fill=(180, 180, 180, int(200 - (16 - size) * 15)))
    
    # Bridge
    bridge_left = left_eye[0] + lens_w//2
    bridge_right = right_eye[0] - lens_w//2
    bridge_y = (left_eye[1] + right_eye[1]) // 2
    bridge_h = int(lens_h * 0.25)
    
    for i in range(max(1, bridge_right - bridge_left)):
        progress = i / max(1, bridge_right - bridge_left)
        shade = int(5 + progress * 15)
        draw.rectangle([bridge_left + i, bridge_y - bridge_h//2,
                       bridge_left + i + 1, bridge_y + bridge_h//2],
                      fill=(shade, shade, shade, 255))
    
    # Temple arms
    for i in range(10):
        alpha = int(255 - i * 15)
        draw.line([left_eye[0] - lens_w//2 - 8, bridge_y + i - 5, 10, bridge_y + i - 5],
                 fill=(0, 0, 0, alpha), width=3)
        draw.line([right_eye[0] + lens_w//2 + 8, bridge_y + i - 5,
                  overlay.size[0] - 10, bridge_y + i - 5],
                 fill=(0, 0, 0, alpha), width=3)
    
    overlay = overlay.resize(img_rgba.size, Image.LANCZOS)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    result = Image.alpha_composite(img_rgba, overlay)
    print("   ✅ Sunglasses complete!")
    return result.convert('RGB')

def add_premium_mask(image):
    """Mask from nose to chin"""
    landmarks = get_facial_landmarks_opencv(image)
    
    if not landmarks:
        w, h = image.size
        landmarks = {
            'nose_tip': (int(w * 0.50), int(h * 0.50)),
            'nose_bridge': (int(w * 0.50), int(h * 0.40)),
            'chin': (int(w * 0.50), int(h * 0.72)),
            'jaw_left': (int(w * 0.20), int(h * 0.65)),
            'jaw_right': (int(w * 0.80), int(h * 0.65)),
            'mouth_bottom': (int(w * 0.50), int(h * 0.60))
        }
    else:
        print("   ✓ Using OpenCV landmarks for mask")
    
    img_rgba = image.convert('RGBA')
    scale = 2
    overlay = Image.new('RGBA', (img_rgba.size[0] * scale, img_rgba.size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    nose_tip = tuple(v * scale for v in landmarks['nose_tip'])
    nose_bridge = tuple(v * scale for v in landmarks['nose_bridge'])
    chin = tuple(v * scale for v in landmarks['chin'])
    jaw_left = tuple(v * scale for v in landmarks['jaw_left'])
    jaw_right = tuple(v * scale for v in landmarks['jaw_right'])
    mouth_bottom = tuple(v * scale for v in landmarks['mouth_bottom'])
    
    # Mask from mid-nose to below chin
    mask_top = int(nose_bridge[1] + (nose_tip[1] - nose_bridge[1]) * 0.5)
    mask_bottom = int(chin[1] + (chin[1] - mouth_bottom[1]) * 0.4)
    
    mask_points = [
        (jaw_left[0] + 50, mask_top + 20),
        (nose_tip[0] - 65, mask_top),
        (nose_tip[0], mask_top - 10),
        (nose_tip[0] + 65, mask_top),
        (jaw_right[0] - 50, mask_top + 20),
        (jaw_right[0] - 20, mask_bottom - 25),
        (chin[0] + 45, mask_bottom),
        (chin[0], mask_bottom + 10),
        (chin[0] - 45, mask_bottom),
        (jaw_left[0] + 20, mask_bottom - 25)
    ]
    
    # Draw mask body
    for y_off in range(-2, 3):
        pts = [(p[0], p[1] + y_off) for p in mask_points]
        draw.polygon(pts, fill=(238 + y_off, 244, 252, int(248 + y_off * 2)))
    
    # Pleats
    mask_h = mask_bottom - mask_top
    for i in range(6):
        pleat_y = mask_top + int(mask_h * (0.10 + i * 0.15))
        pts = [(x, pleat_y + int(8 * math.sin((x - jaw_left[0]) / max(1, jaw_right[0] - jaw_left[0]) * math.pi)))
               for x in range(jaw_left[0] + 60, jaw_right[0] - 60, 15)]
        if len(pts) > 1:
            draw.line(pts, fill=(210, 220, 230, 230), width=4)
            draw.line([(p[0], p[1] + 2) for p in pts], fill=(195, 205, 215, 170), width=2)
    
    # Outline
    for t in range(4, 0, -1):
        draw.polygon(mask_points, outline=(190, 200, 212, int(255 - (4 - t) * 40)), width=t)
    
    # Ear loops
    loop_y = mask_top + int(mask_h * 0.35)
    for side_x, angles in [(jaw_left[0], (90, 270)), (jaw_right[0], (270, 450))]:
        for t in range(6, 0, -1):
            bbox = ([side_x - 90 - t, loop_y - 20, side_x + 50 - t, loop_y + 20] if side_x == jaw_left[0]
                   else [side_x - 50 + t, loop_y - 20, side_x + 90 + t, loop_y + 20])
            draw.arc(bbox, angles[0], angles[1], fill=(228, 235, 245, int(235 - (6 - t) * 20)), width=2)
    
    # Nose strip
    for i in range(8):
        draw.rectangle([nose_tip[0] - 25, mask_top + 8 + i, nose_tip[0] + 25, mask_top + 12 + i],
                      fill=(195 + i * 5, 195 + i * 5, 200 + i * 5, int(230 - i * 15)))
    
    overlay = overlay.resize(img_rgba.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=0.8))
    result = Image.alpha_composite(img_rgba, overlay)
    print("   ✅ Mask complete!")
    return result.convert('RGB')

def add_premium_beard(image):
    """Beard from below mouth to chin"""
    landmarks = get_facial_landmarks_opencv(image)
    
    if not landmarks:
        w, h = image.size
        landmarks = {
            'mouth_bottom': (int(w * 0.50), int(h * 0.60)),
            'chin': (int(w * 0.50), int(h * 0.72)),
            'jaw_left': (int(w * 0.25), int(h * 0.65)),
            'jaw_right': (int(w * 0.75), int(h * 0.65))
        }
    else:
        print("   ✓ Using OpenCV landmarks for beard")
    
    img_rgba = image.convert('RGBA')
    scale = 2
    overlay = Image.new('RGBA', (img_rgba.size[0] * scale, img_rgba.size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    mouth_bottom = tuple(v * scale for v in landmarks['mouth_bottom'])
    chin = tuple(v * scale for v in landmarks['chin'])
    jaw_left = tuple(v * scale for v in landmarks['jaw_left'])
    jaw_right = tuple(v * scale for v in landmarks['jaw_right'])
    
    beard_start_y = mouth_bottom[1] + 15
    beard_end_y = int(chin[1] + (chin[1] - mouth_bottom[1]) * 0.7)
    
    beard_pts = [
        (jaw_left[0] + 70, beard_start_y),
        (jaw_left[0] + 40, beard_start_y + int((beard_end_y - beard_start_y) * 0.2)),
        (jaw_left[0] + 25, chin[1] - 15),
        (jaw_left[0] + 20, beard_end_y - 15),
        (chin[0] - 55, beard_end_y + 25),
        (chin[0], beard_end_y + 35),
        (chin[0] + 55, beard_end_y + 25),
        (jaw_right[0] - 20, beard_end_y - 15),
        (jaw_right[0] - 25, chin[1] - 15),
        (jaw_right[0] - 40, beard_start_y + int((beard_end_y - beard_start_y) * 0.2)),
        (jaw_right[0] - 70, beard_start_y)
    ]
    
    # Base layers
    base_c = (38, 28, 18)
    for layer in range(3):
        pts = [(p[0] + layer * 2, p[1] + layer * 2) for p in beard_pts]
        draw.polygon(pts, fill=(*base_c, int(220 - layer * 15)))
    
    # Hair strands
    np.random.seed(42)
    for _ in range(500):
        x = int(jaw_left[0] + 30 + np.random.random() * (jaw_right[0] - jaw_left[0] - 60))
        y = int(beard_start_y + np.random.random() * (beard_end_y - beard_start_y + 35))
        length = np.random.randint(8, 22)
        angle = np.random.randint(-40, 40)
        
        color_var = np.random.randint(-20, 32)
        hair_c = (min(255, max(0, base_c[0] + color_var)),
                 min(255, max(0, base_c[1] + color_var)),
                 min(255, max(0, base_c[2] + color_var - 4)),
                 np.random.randint(180, 245))
        
        end_x = x + int(length * 0.30 * np.sin(np.radians(angle)))
        end_y = y + int(length * np.cos(np.radians(angle)))
        draw.line([(x, y), (end_x, end_y)], fill=hair_c, width=np.random.choice([1, 1, 1, 2]))
    
    # Highlights
    for _ in range(120):
        x = int(jaw_left[0] + 30 + np.random.random() * (jaw_right[0] - jaw_left[0] - 60))
        y = int(beard_start_y + np.random.random() * (beard_end_y - beard_start_y))
        length = np.random.randint(6, 14)
        draw.line([(x, y), (x + np.random.randint(-4, 4), y + length)],
                 fill=(82, 72, 62, 190), width=1)
    
    overlay = overlay.resize(img_rgba.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=0.4))
    result = Image.alpha_composite(img_rgba, overlay)
    print("   ✅ Beard complete!")
    return result.convert('RGB')

def add_tilt_left(image):
    """Tilt face to the left with dramatic lighting from right"""
    print("   Creating left tilt with lighting...")
    
    # Rotate image 15 degrees counter-clockwise
    img_rgba = image.convert('RGBA')
    rotated = img_rgba.rotate(15, expand=False, fillcolor=(255, 255, 255, 0))
    
    # Add lighting effect - bright from right, shadow on left
    w, h = rotated.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    # Create gradient lighting from right to left
    for x in range(w):
        progress = x / w
        # Right side bright (white/yellow), left side dark (blue shadow)
        if progress > 0.5:  # Right half - warm light
            brightness = int((progress - 0.5) * 2 * 120)
            draw.line([(x, 0), (x, h)], fill=(255, 250, 200, brightness), width=1)
        else:  # Left half - cool shadow
            darkness = int((0.5 - progress) * 2 * 80)
            draw.line([(x, 0), (x, h)], fill=(30, 50, 80, darkness), width=1)
    
    # Apply lighting
    result = Image.alpha_composite(rotated, overlay)
    
    # Enhance contrast for dramatic effect
    enhancer = ImageEnhance.Contrast(result.convert('RGB'))
    result = enhancer.enhance(1.2)
    
    print("   ✅ Left tilt complete!")
    return result

def add_tilt_right(image):
    """Tilt face to the right with dramatic lighting from left"""
    print("   Creating right tilt with lighting...")
    
    # Rotate image 15 degrees clockwise
    img_rgba = image.convert('RGBA')
    rotated = img_rgba.rotate(-15, expand=False, fillcolor=(255, 255, 255, 0))
    
    # Add lighting effect - bright from left, shadow on right
    w, h = rotated.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    # Create gradient lighting from left to right
    for x in range(w):
        progress = x / w
        # Left side bright (white/yellow), right side dark (blue shadow)
        if progress < 0.5:  # Left half - warm light
            brightness = int((0.5 - progress) * 2 * 120)
            draw.line([(x, 0), (x, h)], fill=(255, 250, 200, brightness), width=1)
        else:  # Right half - cool shadow
            darkness = int((progress - 0.5) * 2 * 80)
            draw.line([(x, 0), (x, h)], fill=(30, 50, 80, darkness), width=1)
    
    # Apply lighting
    result = Image.alpha_composite(rotated, overlay)
    
    # Enhance contrast for dramatic effect
    enhancer = ImageEnhance.Contrast(result.convert('RGB'))
    result = enhancer.enhance(1.2)
    
    print("   ✅ Right tilt complete!")
    return result

def add_premium_hat(image):
    """Hat positioned on forehead"""
    landmarks = get_facial_landmarks_opencv(image)
    
    if not landmarks:
        w, h = image.size
        cap_width = int(w * 0.38)
        cap_center_x = int(w * 0.50)
        cap_top = int(h * 0.08)
        cap_bottom = int(h * 0.30)
    else:
        print("   ✓ Using OpenCV landmarks for hat")
        forehead = landmarks['forehead']
        left_eye = landmarks['left_eye_center']
        right_eye = landmarks['right_eye_center']
        
        eye_dist = math.dist(left_eye, right_eye)
        cap_width = int(eye_dist * 1.5)
        cap_center_x = (left_eye[0] + right_eye[0]) // 2
        cap_top = int(forehead[1] - eye_dist * 0.5)
        cap_bottom = int(forehead[1] + eye_dist * 0.15)
    
    img_rgba = image.convert('RGBA')
    scale = 2
    overlay = Image.new('RGBA', (img_rgba.size[0] * scale, img_rgba.size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    cap_width *= scale
    cap_center_x *= scale
    cap_top *= scale
    cap_bottom *= scale
    cap_height = cap_bottom - cap_top
    
    # 3D gradient crown
    for i in range(cap_height):
        progress = i / cap_height
        color = ((60, 100, 235) if progress < 0.20 else
                (45, 80, 210) if progress < 0.40 else
                (35, 65, 195) if progress < 0.70 else
                (20, 45, 145))
        
        width_factor = 0.60 + 0.40 * (1 - abs(progress - 0.5) * 2)
        w_at_level = int(cap_width * width_factor)
        x_off = (cap_width - w_at_level) // 2
        
        draw.rectangle([cap_center_x - cap_width//2 + x_off, cap_top + i,
                       cap_center_x - cap_width//2 + x_off + w_at_level, cap_top + i + 1],
                      fill=(*color, 255))
    
    # Curved bill
    bill_w = int(cap_width * 1.25)
    bill_h = int(cap_height * 0.40)
    
    for i in range(bill_h):
        curve = int((i ** 1.35) * 0.35)
        shade = int(25 - i / bill_h * 15)
        draw.rectangle([cap_center_x - bill_w//2 + curve, cap_bottom + i,
                       cap_center_x + bill_w//2 - curve, cap_bottom + i + 1],
                      fill=(shade, 50 + shade, 180 + shade, 250))
    
    # Shadow line
    draw.line([cap_center_x - bill_w//2 + 25, cap_bottom + 5,
              cap_center_x + bill_w//2 - 25, cap_bottom + 5],
             fill=(0, 0, 0, 120), width=4)
    
    overlay = overlay.resize(img_rgba.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=0.6))
    result = Image.alpha_composite(img_rgba, overlay)
    print("   ✅ Hat complete!")
    return result.convert('RGB')

@app.route('/augment', methods=['POST'])
def augment():
    try:
        data = request.json
        img_base64 = data.get('img_base64')
        augmentation_types = data.get('augmentations', ['all'])
        
        if not img_base64:
            return jsonify({'error': 'img_base64 required'}), 400
        
        original = base64_to_image(img_base64)
        print(f"\n📸 Processing: {original.size}")
        
        results = []
        augmentations = {
            'sunglasses': add_premium_sunglasses,
            'mask': add_premium_mask,
            'beard': add_premium_beard,
            'tilt_left': add_tilt_left,
            'tilt_right': add_tilt_right
        }
        
        if 'all' in augmentation_types:
            augmentation_types = ['original', 'sunglasses', 'mask', 'beard', 'tilt_left', 'tilt_right']
        
        for aug_type in augmentation_types:
            if aug_type == 'original':
                results.append({'augmentation_type': 'original', 'img_base64': img_base64, 'success': True})
                print("✅ Original")
                continue
            
            if aug_type in augmentations:
                try:
                    print(f"\n🎨 Adding {aug_type}...")
                    augmented = augmentations[aug_type](original.copy())
                    results.append({
                        'augmentation_type': aug_type,
                        'img_base64': image_to_base64(augmented),
                        'success': True
                    })
                except Exception as e:
                    print(f"❌ {aug_type} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({'augmentation_type': aug_type, 'error': str(e), 'success': False})
        
        success = sum(1 for r in results if r['success'])
        print(f"\n📊 Complete: {success}/{len(results)} successful\n")
        
        return jsonify({
            'status': 'success',
            'total_augmentations': len(results),
            'successful': success,
            'results': results
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': 'opencv-v5'})

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 FREE Face Augmentation - NO DOWNLOADS!")
    print("=" * 70)
    print("✅ Uses OpenCV's built-in Haar Cascades")
    print("📦 Install: pip install opencv-python pillow numpy")
    print("🔒 No external files needed - completely secure!")
    print("📍 http://localhost:3001")
    print("=" * 70 + "\n")
    app.run(host='0.0.0.0', port=3001, debug=True)