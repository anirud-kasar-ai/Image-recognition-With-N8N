"""Small test harness to run `detect_sunglasses_advanced` on images in `images/` and save debug outputs.

Usage:
    python scripts/test_sunglasses.py --debug --out-dir debug_out
"""
import os
import argparse
import cv2
import json

from multi_face import detect_sunglasses_advanced

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
OUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'frames_output', 'sunglasses_debug')

os.makedirs(OUT_BASE, exist_ok=True)


def annotate_and_save(img_path, res, out_dir):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    # Draw full result text
    txt = f"Sunglasses: {res['sunglasses']} conf={res['confidence']:.2f}"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if res['sunglasses'] else (0, 0, 255), 2)

    # Draw eye band rectangle used by detector
    y1 = int(h * 0.08)
    y2 = int(h * 0.62)
    x1 = int(w * 0.03)
    x2 = int(w * 0.97)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Add a JSON dump of details at the bottom
    details_txt = json.dumps(res['details'])
    # Keep it short for display
    cv2.putText(img, details_txt[:200], (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--out-dir', default=OUT_BASE)
    parser.add_argument('--min-confidence', type=float, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    image_files = [os.path.join(IMAGES_DIR, f) for f in sorted(os.listdir(IMAGES_DIR)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print('No images found in', IMAGES_DIR)
        return

    for p in image_files:
        img = cv2.imread(p)
        if img is None:
            print('Failed to read', p)
            continue
        res = detect_sunglasses_advanced(img, debug=args.debug)
        if args.min_confidence is not None:
            res['sunglasses'] = bool(res['confidence'] >= args.min_confidence)
        out_path = annotate_and_save(p, res, out_dir)
        print(p, '->', out_path, '->', res)


if __name__ == '__main__':
    main()
