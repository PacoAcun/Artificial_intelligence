import numpy as np
import xml.etree.ElementTree as ET
import cv2

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box: [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    
    return inter_area / union_area

def parse_xml_annotation(xml_path):
    """
    Parses an XML file to extract bounding box coordinates.
    Returns: [x, y, w, h] (assuming single object or taking the first one)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find the first object (assuming one plate per image for now)
        obj = root.find('object')
        if obj is None:
            return None
            
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        return [xmin, ymin, xmax - xmin, ymax - ymin]
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None
