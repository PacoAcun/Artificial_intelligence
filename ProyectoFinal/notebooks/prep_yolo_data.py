import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


CSV_PATH = '../data/processed/train_detection.csv'
YOLO_DIR = '../data/yolo_dataset'


def resolve_image_path(img_path: str) -> str:
    """Resolve an image path stored in the CSV.

    Handles absolute paths and common relative variants.
    """
    if os.path.exists(img_path):
        return img_path

    # Try relative to project root
    alt_path = os.path.join('..', img_path)
    if os.path.exists(alt_path):
        return alt_path

    # Try data/raw with the filename
    alt_path = os.path.join('../data/raw', os.path.basename(img_path))
    if os.path.exists(alt_path):
        return alt_path

    raise FileNotFoundError(f'Image not found for path: {img_path}')


def convert_to_yolo(row, split: str) -> bool:
    """Convert one bounding box from (x, y, w, h) to YOLO format.

    Output: class x_center y_center width height (all normalized to [0, 1]).
    """
    try:
        img_path = resolve_image_path(row['image_path'])
    except FileNotFoundError as e:
        print(e)
        return False

    x, y, w, h = row['x'], row['y'], row['w'], row['h']

    # Read image to get dimensions
    img = cv2.imread(img_path)
    if img is None:
        print(f'Warning: could not read image: {img_path}')
        return False

    img_h, img_w = img.shape[:2]

    # Convert to YOLO format
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    # Copy image to YOLO directory
    img_name = os.path.basename(img_path)
    dest_img = os.path.join(YOLO_DIR, 'images', split, img_name)
    os.makedirs(os.path.dirname(dest_img), exist_ok=True)
    shutil.copy(img_path, dest_img)

    # Create label file
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(YOLO_DIR, 'labels', split, label_name)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    # Class 0 = license_plate
    with open(label_path, 'w') as f:
        f.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

    return True


def main() -> None:
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

    print('Created YOLO directory structure')

    if not os.path.exists(CSV_PATH):
        print(f'Error: CSV file not found at {CSV_PATH}')
        return

    df = pd.read_csv(CSV_PATH)
    print(f'Total samples: {len(df)}')

    # Split into train/val (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f'Train: {len(train_df)}, Val: {len(val_df)}')

    # Convert training set
    print('Converting training set...')
    train_success = 0
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        if convert_to_yolo(row, 'train'):
            train_success += 1

    print(f'Successfully converted {train_success}/{len(train_df)} training samples')

    # Convert validation set
    print('Converting validation set...')
    val_success = 0
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        if convert_to_yolo(row, 'val'):
            val_success += 1

    print(f'Successfully converted {val_success}/{len(val_df)} validation samples')

    # Create data.yaml configuration file
    yaml_content = f"""# YOLOv8 License Plate Detection Dataset
path: {os.path.abspath(YOLO_DIR)}
train: images/train
val: images/val

# Classes
nc: 1  # number of classes
names: ['license_plate']
"""

    yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f'Created {yaml_path}')
    print('\nDataset ready for YOLOv8 training!')


if __name__ == '__main__':
    main()
