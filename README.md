# Flower Classification with Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
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

- **Total Images:** ~3,670 labeled photos
- **Classes:** 5 flower types
- **Image Format:** JPEG, various sizes
- **Train/Test Split:** 80/20

| Class      | Training | Test |
|------------|----------|------|
| Daisy      | ~525     | ~131 |
| Dandelion  | ~720     | ~180 |
| Rose       | ~560     | ~140 |
| Sunflower  | ~560     | ~140 |
| Tulip      | ~640     | ~160 |

## Project Structure

```
flower-classification-capstone/
├── README.md                 # Project documentation
├── notebooks/
│   └── eda_and_training.ipynb  # EDA + model experiments
├── src/
│   ├── train.py              # Model training script
│   └── predict.py            # Prediction/inference script
├── models/                   # Saved model artifacts
├── data/                     # Dataset (downloaded separately)
├── docker/
│   └── Dockerfile            # Container definition
├── requirements.txt          # Python dependencies
├── Pipfile                   # Pipenv dependencies
└── tests/                    # Test scripts
```

## Model Approach

### Baseline
- Simple CNN built from scratch (Conv2D → MaxPool → Dense)

### Transfer Learning (Final Model)
- **Base Model:** MobileNet V2 (pretrained on ImageNet)
- **Strategy:** Feature extraction + fine-tuning top layers
- **Input Size:** 224×224×3
- **Optimizer:** Adam with learning rate scheduling
- **Regularization:** Dropout, data augmentation

### Model Comparison

| Model | Test Accuracy | Training Time |
|-------|---------------|---------------|
| Baseline CNN | ~65% | ~5 min |
| MobileNetV2 (frozen) | ~88% | ~3 min |
| MobileNetV2 (fine-tuned) | ~92% | ~10 min |

*(Results will be updated after training)*

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/flower-classification-capstone.git
cd flower-classification-capstone
```

### 2. Set Up Environment

**Option A: Using Pipenv (recommended)**
```bash
pip install pipenv
pipenv install
pipenv shell
```

**Option B: Using pip**
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

### 4. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Train the model with the best hyperparameters
- Save the model to `models/flower_classifier.keras`

### 5. Run the Web Service

```bash
python src/predict.py
```

The API will start at `http://localhost:9696`

### 6. Test a Prediction

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/flower.jpg"}'
```

Or with a local file:
```bash
curl -X POST http://localhost:9696/predict \
  -F "image=@path/to/flower.jpg"
```

**Expected Response:**
```json
{
  "prediction": "rose",
  "confidence": 0.94,
  "probabilities": {
    "daisy": 0.02,
    "dandelion": 0.01,
    "rose": 0.94,
    "sunflower": 0.02,
    "tulip": 0.01
  }
}
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
  -d '{"image_url": "https://example.com/sunflower.jpg"}'
```

## Cloud Deployment

*(Optional - for bonus points)*

Instructions for deploying to cloud/Kubernetes will be added here.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify a flower image |
| `/health` | GET | Health check (returns 200 OK) |

## Notebooks

- **[EDA and Training](notebooks/eda_and_training.ipynb)**: Exploratory data analysis, model experimentation, and hyperparameter tuning

## Key Findings

*(To be updated after EDA)*

1. **Class Distribution:** Slightly imbalanced, dandelion has the most images
2. **Image Quality:** Varied lighting, backgrounds, and flower orientations
3. **Challenges:** Some classes (tulip/rose) share similar colors

## Evaluation Metrics

- **Primary Metric:** Accuracy (balanced dataset)
- **Secondary:** Per-class precision/recall, confusion matrix

## Limitations & Future Work

- Model trained on 5 flower types only
- Performance may vary with low-quality or unusual images
- Future: Expand to more flower species, add mobile app integration

## Technologies Used

- Python 3.11
- TensorFlow 2.x / Keras
- Flask
- Docker
- NumPy, Pandas, Matplotlib, Seaborn

## Author

**Michael** - ML Zoomcamp 2025 Capstone Project

## License

This project is licensed under the MIT License.

## Acknowledgments

- [DataTalks.Club](https://datatalks.club/) for the ML Zoomcamp course
- TensorFlow team for the flowers dataset
- MobileNet V2 paper authors
