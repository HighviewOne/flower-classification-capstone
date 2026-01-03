# Flower Classification with Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

An image classification model that identifies 5 types of flowers using transfer learning with MobileNet V2. Built as a capstone project for [ML Zoomcamp 2025](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem Description

Flower identification is a common challenge for gardeners, botanists, and nature enthusiasts. This project builds an AI-powered classifier that can identify flowers from photographs, making it easier to:

- Identify unknown flowers while hiking or gardening
- Assist in botanical research and cataloging
- Power mobile apps for plant identification

The model classifies images into **5 flower categories**:
- 🌼 Daisy
- 🌻 Sunflower
- 🌷 Tulip
- 🌹 Rose
- 🌾 Dandelion

## Dataset

**Source:** [TensorFlow Flowers Dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)

- **Total Images:** 3,670 labeled photos
- **Classes:** 5 flower types
- **Image Format:** JPEG, various sizes
- **Train/Val Split:** 80/20

| Class      | Total | Training | Validation |
|------------|-------|----------|------------|
| Daisy      | 633   | 526      | 107        |
| Dandelion  | 898   | 707      | 191        |
| Roses      | 641   | 522      | 119        |
| Sunflowers | 699   | 564      | 135        |
| Tulips     | 799   | 617      | 182        |
| **Total**  | **3,670** | **2,936** | **734** |

### Class Distribution

The dataset is relatively balanced, with a class imbalance ratio of ~1.42 (dandelion has the most images, daisy the fewest).

## Project Structure

```
flower-classification-capstone/
├── README.md                 # Project documentation
├── notebooks/
│   └── eda_and_training.ipynb  # EDA + model experiments
├── src/
│   ├── download_data.py      # Dataset download script
│   ├── train.py              # Model training script
│   └── predict.py            # Flask prediction service
├── models/                   # Saved model artifacts
│   ├── flower_classifier.keras
│   └── class_names.txt
├── data/                     # Dataset (downloaded separately)
├── docker/
│   └── Dockerfile            # Container definition
├── tests/
│   └── test_service.py       # API test script
├── requirements.txt          # Python dependencies
└── Pipfile                   # Pipenv dependencies
```

## Model Approach

### Baseline: Simple CNN
A basic convolutional neural network built from scratch:
- 3 Conv2D + MaxPooling blocks
- Dense layer with dropout
- **Result:** Severe overfitting (96% train, 67% validation)

### Transfer Learning: MobileNet V2
Using a pre-trained MobileNet V2 (ImageNet weights) as a feature extractor:
- Frozen base model + custom classification head
- Data augmentation (flip, rotation, zoom)
- GlobalAveragePooling + Dropout + Dense(5)
- **Result:** 88% validation accuracy

### Fine-Tuning
Unfreezing the last 30 layers of MobileNet V2 for fine-tuning:
- Lower learning rate (1e-5)
- 5 additional epochs
- **Result:** 88.3% validation accuracy (best model)

## Training Results

| Model | Train Accuracy | Val Accuracy | Notes |
|-------|----------------|--------------|-------|
| Baseline CNN | 96% | 67% | Overfitting |
| MobileNetV2 (frozen) | 90% | 88% | Transfer learning |
| MobileNetV2 (fine-tuned) | 90% | **88.3%** | **Selected model** |

### Per-Class Performance (Final Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Daisy | 0.83 | 0.95 | 0.89 | 107 |
| Dandelion | 0.95 | 0.91 | 0.93 | 191 |
| Roses | 0.77 | 0.92 | 0.84 | 119 |
| Sunflowers | 0.93 | 0.84 | 0.89 | 135 |
| Tulips | 0.90 | 0.82 | 0.86 | 182 |
| **Overall** | **0.88** | **0.88** | **0.88** | **734** |

### Key Observations

1. **Dandelion** has the highest precision (95%) - distinctive yellow color and shape
2. **Roses** has the lowest precision (77%) - sometimes confused with tulips due to similar colors
3. **Daisy** has the highest recall (95%) - white petals with yellow center are easy to identify
4. **Tulips** has the lowest recall (82%) - varied colors cause some confusion with roses

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/HighviewOne/flower-classification-capstone.git
cd flower-classification-capstone
```

### 2. Set Up Environment

**Option A: Using Conda (recommended for this project)**
```bash
conda activate MLZoomCamp_env
pip install -r requirements.txt
```

**Option B: Using Pipenv**
```bash
pip install pipenv
pipenv install
pipenv shell
```

**Option C: Using pip with venv**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Dataset

```bash
python src/download_data.py
```

Or manually:
```bash
cd data
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar -xzf flower_photos.tgz
```

### 4. Train the Model (Optional)

The trained model is already included. To retrain:

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Train the MobileNetV2 transfer learning model
- Save the model to `models/flower_classifier.keras`

### 5. Run the Web Service

```bash
python src/predict.py
```

The API will start at `http://localhost:9696`

### 6. Test a Prediction

**Using an image URL:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/800px-Sunflower_sky_backdrop.jpg"}'
```

**Using a local file:**
```bash
curl -X POST http://localhost:9696/predict \
  -F "image=@path/to/flower.jpg"
```

**Expected Response:**
```json
{
  "prediction": "sunflowers",
  "confidence": 0.9823,
  "probabilities": {
    "daisy": 0.0012,
    "dandelion": 0.0034,
    "roses": 0.0045,
    "sunflowers": 0.9823,
    "tulips": 0.0086
  }
}
```

### 7. Run the Test Suite

```bash
python tests/test_service.py
```

## Docker

### Build the Container

```bash
docker build -t flower-classifier -f docker/Dockerfile .
```

### Run the Container

```bash
docker run -it -p 9696:9696 flower-classifier
```

### Test the Containerized Service

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/800px-Sunflower_sky_backdrop.jpg"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify a flower image |
| `/health` | GET | Health check (returns 200 OK) |

### Request Formats

The `/predict` endpoint accepts three input formats:

1. **JSON with image URL:**
   ```json
   {"image_url": "https://example.com/flower.jpg"}
   ```

2. **JSON with base64-encoded image:**
   ```json
   {"image_base64": "iVBORw0KGgo..."}
   ```

3. **Multipart form with file upload:**
   ```bash
   curl -F "image=@flower.jpg" http://localhost:9696/predict
   ```

## Notebooks

- **[EDA and Training](notebooks/eda_and_training.ipynb)**: Complete exploratory data analysis, model experimentation, and hyperparameter tuning

### EDA Highlights

- **Image sizes vary** from 240×180 to 4000×3000 pixels (resized to 224×224 for training)
- **Aspect ratios** are mostly close to 1.0 (square-ish images)
- **Data augmentation** (random flip, rotation, zoom) helps prevent overfitting

## Technologies Used

- **Python 3.11**
- **TensorFlow 2.20 / Keras** - Deep learning framework
- **MobileNet V2** - Pre-trained CNN for transfer learning
- **Flask** - Web service framework
- **Docker** - Containerization
- **NumPy, Pandas** - Data manipulation
- **Matplotlib, Seaborn** - Visualization
- **scikit-learn** - Metrics and evaluation

## Limitations & Future Work

### Current Limitations
- Model trained on only 5 flower types
- Performance may vary with low-quality, blurry, or unusual angle images
- Flowers with similar colors (roses/tulips) can be confused

### Future Improvements
- Expand to more flower species (10-20 classes)
- Add confidence thresholding to reject uncertain predictions
- Implement model versioning and A/B testing
- Deploy to cloud (AWS/GCP) with auto-scaling
- Build a mobile app with camera integration

## Author

**Michael** - ML Zoomcamp 2025 Capstone Project

- GitHub: [@HighviewOne](https://github.com/HighviewOne)

## License

This project is licensed under the MIT License.

## Acknowledgments

- [DataTalks.Club](https://datatalks.club/) for the ML Zoomcamp course
- TensorFlow team for the flowers dataset and MobileNet V2
- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381) - Sandler et al., 2018
