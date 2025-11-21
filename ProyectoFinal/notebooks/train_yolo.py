from ultralytics import YOLO
import os
import shutil

# Paths
DATA_YAML = os.path.abspath('../data/yolo_dataset/data.yaml')
MODEL_SAVE_DIR = '../models'
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# Training parameters
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL_SIZE = 'yolov8n.pt' 

print(f'Starting training with {MODEL_SIZE}...')
print(f'Data: {DATA_YAML}')

# Load pre-trained model
model = YOLO(MODEL_SIZE)

# Train the model
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='yolo_plate_detector',
    patience=10,
    save=True,
    device='cpu', # Force CPU to avoid CUDA errors
    exist_ok=True # Overwrite existing run
)

# Save the best model
best_model_path = 'runs/detect/yolo_plate_detector/weights/best.pt'
destination = os.path.join(MODEL_SAVE_DIR, 'yolo_plate_detector.pt')

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, destination)
    print(f'✓ Best model saved to: {destination}')
else:
    print(f'⚠️ Best model not found at {best_model_path}')
    # Fallback: try to find where it saved
    print("Listing runs/detect/yolo_plate_detector/weights/:")
    if os.path.exists('runs/detect/yolo_plate_detector/weights'):
        print(os.listdir('runs/detect/yolo_plate_detector/weights'))
