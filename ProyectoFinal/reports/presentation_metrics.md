# ANPR Project - Model Training Metrics & Architecture

## 1. License Plate Detector (YOLOv8)

### Model Configuration

- **Model Architecture:** YOLOv8n (Nano)
- **Pre-trained Weights:** COCO (fine-tuned for 1 class)
- **Input Resolution:** 640x640
- **Batch Size:** 16
- **Epochs:** 20
- **Optimizer:** Auto (AdamW)

### Performance Metrics (Validation Set)

> [!IMPORTANT]
> The model achieved exceptional detection performance with a mean Average Precision (mAP@0.5) of **98.1%**.

| Metric             | Value      | Description                                                                                       |
| :----------------- | :--------- | :------------------------------------------------------------------------------------------------ |
| **mAP @ 0.5**      | **0.9810** | Mean Average Precision at IoU threshold 0.5. Indicates very high reliability in detecting plates. |
| **mAP @ 0.5:0.95** | **0.6571** | Average mAP across IoU thresholds from 0.5 to 0.95. Shows good localization accuracy.             |
| **Precision**      | **0.9564** | High precision means very few false positives (background detected as plates).                    |
| **Recall**         | **0.9243** | High recall means the model misses very few actual plates.                                        |

### Training Observations

- **Convergence:** The model converged quickly, with significant loss reduction in the first 5 epochs.
- **Speed:** Inference speed is approximately **1.2ms per image** (pre-process + inference + post-process) on GPU, making it suitable for real-time applications.

---

## 2. OCR Character Recognition (Custom CNN)

### Model Configuration

- **Architecture:** Custom CNN (Sequential)
  - 2x Convolutional Blocks (Conv2D + MaxPooling)
  - Flatten -> Dense (128) -> Dropout (0.5) -> Output
- **Input Size:** 32x32 (Grayscale)
- **Classes:** 35 (0-9, A-Z)
  - _Note: The letter 'O' is excluded to prevent confusion with the digit '0'._
- **Batch Size:** 32
- **Epochs:** 20

### Performance Metrics (Validation Set)

> [!NOTE]
> The OCR model reached near-perfect accuracy on the validation dataset.

| Metric                  | Value      |
| :---------------------- | :--------- |
| **Validation Accuracy** | **99.94%** |
| **Validation Loss**     | **0.0043** |
| **Training Accuracy**   | **99.33%** |
| **Training Loss**       | **0.0204** |

### Architecture Details

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        320
 max_pooling2d (MaxPooling2D) (None, 15, 15, 32)       0
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18,496
 max_pooling2d_1 (MaxPooling2D) (None, 6, 6, 64)       0
 flatten (Flatten)           (None, 2304)              0
 dense (Dense)               (None, 128)               295,040
 dropout (Dropout)           (None, 128)               0
 dense_1 (Dense)             (None, 35)                4,515
=================================================================
Total params: 318,371
```

### Key Design Decisions

1.  **Grayscale Input:** Reduces computational complexity while retaining sufficient geometric information for character recognition.
2.  **Dropout (0.5):** Applied before the final layer to prevent overfitting, ensuring the model generalizes well to new fonts or slightly distorted characters.
3.  **Class Handling:** Explicitly handling 35 classes (0-9, A-Z excluding 'O') aligns with standard ANPR practices to reduce ambiguity.
