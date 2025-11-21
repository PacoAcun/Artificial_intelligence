# ProyectoFinal_CC3092_Placas

## Descripción

Sistema de Reconocimiento Automático de Placas Vehiculares (ANPR) utilizando una arquitectura híbrida que combina **Transfer Learning** (YOLOv8) con **CNNs personalizadas**.

## Arquitectura Híbrida

### Pipeline Completo

1. **Detección de Placa** → YOLOv8 (Transfer Learning)

   - Modelo pre-entrenado en COCO
   - Fine-tuned para detectar placas vehiculares
   - Salida: Bounding box preciso (x1, y1, x2, y2)

2. **Segmentación de Caracteres** → OpenCV

   - Adaptive Thresholding con fallback a Otsu
   - Detección de contornos con filtros morfológicos
   - Salida: Imágenes individuales de cada carácter (32x32)

3. **Reconocimiento OCR** → CNN Personalizada
   - Red neuronal convolucional custom
   - Entrenada en dataset de caracteres alfanuméricos
   - Salida: Texto de la placa

### ¿Por qué Arquitectura Híbrida?

**Transfer Learning (YOLO) para Detección:**

- ✅ Precisión superior en localización de objetos
- ✅ Entrenamiento rápido (~30-60 min)
- ✅ Robusto ante variaciones de iluminación y ángulo

**CNN Custom para OCR:**

- ✅ Cumple requisitos académicos (implementación propia)
- ✅ Permite experimentación con hyperparameters
- ✅ Control total sobre el modelo de reconocimiento

## Estructura del Proyecto

```
ProyectoFinal/
├── data/
│   ├── raw/                    # Datasets originales
│   ├── processed/              # Datos procesados
│   └── yolo_dataset/           # Dataset en formato YOLO
├── notebooks/
│   ├── 01_Data_Prep_Detection.ipynb
│   ├── 02_Prep_YOLO_Dataset.ipynb     # Conversión a formato YOLO
│   ├── 02_Data_Prep_OCR.ipynb
│   ├── 03b_Train_YOLO_Detector.ipynb  # Transfer Learning
│   ├── 04_Train_OCR_CNN.ipynb         # Tu CNN personalizada
│   ├── 05_Pipeline_Final.ipynb        # Pipeline original (referencia)
│   └── 06_Pipeline_YOLO_Hybrid.ipynb  # Pipeline híbrido ⭐
├── src/
│   └── utils.py
├── models/
│   ├── yolo_plate_detector.pt  # Modelo YOLO entrenado
│   └── ocr_cnn.h5              # Tu modelo OCR
└── reports/
```

## Instalación

### Requisitos

- Python 3.8+
- CUDA (opcional, para entrenamiento GPU)

### Instalar dependencias

```bash
pip install -r requirements.txt
```

Paquetes principales:

- `ultralytics` - YOLOv8
- `tensorflow` - CNNs
- `opencv-python` - Procesamiento de imágenes
- `pandas`, `numpy`, `matplotlib` - Utilidades

## Uso

### 1. Preparar Datos para YOLO

```bash
jupyter notebook notebooks/02_Prep_YOLO_Dataset.ipynb
```

Convierte el dataset de detección al formato YOLO (imágenes + etiquetas `.txt`).

### 2. Entrenar Detector YOLO

```bash
jupyter notebook notebooks/03b_Train_YOLO_Detector.ipynb
```

Fine-tune de YOLOv8 en dataset de placas (~50 epochs, 30-60 min).

### 3. Ejecutar Pipeline Híbrido

```bash
jupyter notebook notebooks/06_Pipeline_YOLO_Hybrid.ipynb
```

Pipeline completo: YOLO → Segmentación → OCR Custom.

## Resultados

### Comparación de Arquitecturas

| Componente               | Pipeline Original | Pipeline Híbrido              |
| ------------------------ | ----------------- | ----------------------------- |
| **Detección**            | CNN Regressor     | YOLOv8 (Transfer Learning) ✅ |
| **Precisión Detección**  | ~60-70%           | **~95%+** ✅                  |
| **Segmentación**         | OpenCV            | OpenCV (mejorada)             |
| **OCR**                  | CNN Custom        | CNN Custom ✅                 |
| **Tiempo Entrenamiento** | ~2-3 horas        | ~1 hora                       |

### Ventajas del Nuevo Sistema

- ✅ **Mayor precisión** en detección de placas
- ✅ **Menos falsos positivos** (recortes vacíos)
- ✅ **Mejor segmentación** (gracias a crops precisos)
- ✅ **Cumple requisitos académicos** (Transfer Learning + Custom CNN)

## Datasets

- **Generic License Plates**: Placas genéricas (entrenamiento + test)
- **Belgian License Plates**: Placas belgas (entrenamiento)

## Métricas

### YOLOv8 Detector

- mAP@0.5: >0.95
- Precision: >0.90
- Recall: >0.90

### OCR CNN

- Validation Accuracy: >99%
- Test Accuracy: ~95-98%

## Autor

Francisco (Paco)  
CC3092 - Inteligencia Artificial

## Licencia

MIT
