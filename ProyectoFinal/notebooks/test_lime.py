import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
import random

# Paths
MODEL_PATH = '../models/ocr_cnn.h5'
DATA_DIR = '../data/processed/train_ocr'

# Load Model
print("Loading model...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load Class Names
if os.path.exists(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
    print(f"✓ Classes ({len(class_names)}): {class_names}")
else:
    print("⚠️ Data directory not found. Using default alphanumeric classes.")
    class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

def predict_fn(images):
    """
    Wrapper function for LIME.
    Args:
        images: List of RGB images (N, 32, 32, 3)
    Returns:
        probs: Prediction probabilities (N, num_classes)
    """
    gray_images = []
    for img in images:
        # Convert to grayscale
        if img.shape[-1] == 3:
            g = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2GRAY)
        else:
            g = img
        
        # Resize if necessary
        if g.shape[:2] != (32, 32):
            g = cv2.resize(g, (32, 32))
            
        # Normalize
        if g.max() > 1.0:
            g = g / 255.0
            
        g = np.expand_dims(g, axis=-1)
        gray_images.append(g)
        
    batch = np.array(gray_images)
    return model.predict(batch, verbose=0)

# Initialize LIME Explainer
print("Initializing LIME explainer...")
explainer = lime_image.LimeImageExplainer()

# Pick a sample image
sample_img_path = None
if os.path.exists(DATA_DIR):
    # Try to find a valid image
    for _ in range(10):
        cls = random.choice(class_names)
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir) and len(os.listdir(cls_dir)) > 0:
            img_name = random.choice(os.listdir(cls_dir))
            sample_img_path = os.path.join(cls_dir, img_name)
            break

if sample_img_path is None:
    # Create a dummy image if no data found
    print("Creating dummy image...")
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.putText(img, 'A', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    sample_img_path = "dummy_test.png"
    cv2.imwrite(sample_img_path, img)

print(f"Explaining image: {sample_img_path}")

# Load and preprocess image for LIME (RGB)
img = cv2.imread(sample_img_path)
img = cv2.resize(img, (32, 32))

# Explain
print("Generating explanation...")
explanation = explainer.explain_instance(
    img.astype('double'), 
    predict_fn, 
    top_labels=5, 
    hide_color=0, 
    num_samples=100
)

print("✓ Explanation generated successfully")

# Save visualization
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)

plt.figure()
plt.imshow(mark_boundaries(temp / 255.0 + 0.5, mask))
plt.title("LIME Explanation")
plt.savefig('lime_test_result.png')
print("✓ Visualization saved to lime_test_result.png")
