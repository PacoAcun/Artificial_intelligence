import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Config
DEBUG_DIR = 'debug_output'
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

# Load Models
print("Loading models...")
detector = load_model('../models/detector_cnn.h5', compile=False)
ocr_model = load_model('../models/ocr_cnn.h5', compile=False)

# Class names
class_names = sorted(os.listdir('../data/processed/train_ocr')) if os.path.exists('../data/processed/train_ocr') else []
print(f'Classes: {class_names}')

def find_candidates(thresh, h_plate, debug_img=None):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    print(f"Found {len(contours)} contours.")
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = h / w
        # Relaxed filter: ratio > 0.1, h > 0.2
        if h > h_plate * 0.2 and h < h_plate * 0.95 and ratio > 0.1:
            candidates.append((x, y, w, h))
            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        else:
            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            print(f"Rejected: x={x}, y={y}, w={w}, h={h} (valid h: {h_plate*0.2:.1f}-{h_plate*0.95:.1f}), ratio={ratio:.2f}")
    return candidates

def segment_characters(plate_img, img_name):
    h_plate, w_plate = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Try Adaptive Thresholding
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel)
    
    debug_img_adapt = plate_img.copy()
    candidates_adapt = find_candidates(thresh_adapt, h_plate, debug_img_adapt)
    
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_thresh_adapt.png"), thresh_adapt)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_contours_adapt.png"), debug_img_adapt)
    
    final_candidates = candidates_adapt
    used_method = "Adaptive"
    
    # 2. Fallback to Otsu if few or too many candidates
    if len(candidates_adapt) < 4 or len(candidates_adapt) > 12:
        print(f"Adaptive candidates: {len(candidates_adapt)}. Trying Otsu...")
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        
        debug_img_otsu = plate_img.copy()
        candidates_otsu = find_candidates(thresh_otsu, h_plate, debug_img_otsu)
        
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_thresh_otsu.png"), thresh_otsu)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_contours_otsu.png"), debug_img_otsu)
        
        # Use Otsu if it gives a more reasonable number (4-10)
        if 4 <= len(candidates_otsu) <= 12:
            final_candidates = candidates_otsu
            used_method = "Otsu"
        elif len(candidates_otsu) > len(candidates_adapt) and len(candidates_adapt) < 4:
             # If adaptive had few, and otsu has more (even if not > 4), take otsu
             final_candidates = candidates_otsu
             used_method = "Otsu"
        elif len(candidates_otsu) < len(candidates_adapt) and len(candidates_adapt) > 12:
             # If adaptive had many, and otsu has fewer, take otsu
             final_candidates = candidates_otsu
             used_method = "Otsu"
            
    print(f"Used Segmentation Method: {used_method} (Candidates: {len(final_candidates)})")

    # Sort left to right
    bounding_boxes = sorted(final_candidates, key=lambda b: b[0])
    
    char_rois = []
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Extract from gray image (not thresholded) for recognition? 
        # Usually better to use gray or re-thresholded ROI.
        # Let's use gray as in original code.
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (32, 32))
        char_rois.append(roi)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_char_{i}.png"), roi)
        
    return char_rois

def predict_plate(image_path):
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing {img_name}...")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return
    
    h_orig, w_orig = img.shape[:2]
    
    # Pre-procesamiento para el modelo
    img_resized = cv2.resize(img, (224, 224))
    input_img = np.expand_dims(img_resized / 255.0, axis=0)
    
    # Predicción (x, y, w, h) normalizados
    pred = detector.predict(input_img, verbose=0)[0]
    
    # --- CORRECCIÓN DE COORDENADAS ---
    x_start = int(pred[0] * w_orig)
    y_start = int(pred[1] * h_orig)
    plate_w = int(pred[2] * w_orig)
    plate_h = int(pred[3] * h_orig)
    
    x_end = x_start + plate_w
    y_end = y_start + plate_h
    
    print(f"Raw Prediction: {pred}")
    print(f"Coords: x={x_start}, y={y_start}, w={plate_w}, h={plate_h}")
    
    # Save raw detection
    debug_img_raw = img.copy()
    cv2.rectangle(debug_img_raw, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_detection_raw.png"), debug_img_raw)
    
    # --- PROTECCIONES DE SEGURIDAD ---
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(w_orig, x_end)
    y_end = min(h_orig, y_end)
    
    if x_end <= x_start or y_end <= y_start:
        print(f"⚠️ Advertencia: Detección inválida. Usando imagen completa.")
        x_start, y_start, x_end, y_end = 0, 0, w_orig, h_orig

    # Recorte
    plate_img = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_plate_crop.png"), plate_img)
    
    # 3. Segmentación y OCR
    char_imgs = segment_characters(plate_img, img_name)
    plate_text = ""
    for char_img in char_imgs:
        input_char = np.expand_dims(char_img / 255.0, axis=0)
        input_char = np.expand_dims(input_char, axis=-1)
        probs = ocr_model.predict(input_char, verbose=0)
        idx = np.argmax(probs)
        if idx < len(class_names):
            plate_text += class_names[idx]

    print(f"Predicted Text: {plate_text}")
    
    # Save final result
    img_box = img.copy()
    cv2.rectangle(img_box, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
    cv2.putText(img_box, plate_text, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{img_name}_result.png"), img_box)

# Run on test samples
TEST_DIR = '../data/processed/test_samples'
if os.path.exists(TEST_DIR):
    samples = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png'))]
    # Pick a few samples
    selected_samples = samples[:3] # First 3
    if 'images10.jpg' in samples:
        selected_samples.append('images10.jpg')
    
    for s in set(selected_samples): # Use set to avoid duplicates
        predict_plate(os.path.join(TEST_DIR, s))
else:
    print("Test directory not found.")
