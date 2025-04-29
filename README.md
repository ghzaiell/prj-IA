# Chocolate Type Classifier

A machine learning project that classifies images of chocolate as either Dark or White chocolate using a Random Forest classifier.

## Project Overview

This project includes:
- Image classification model training
- Web interface for predictions
- Command-line testing tool

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Project Structure:
- `train.py` - Trains the model on chocolate images
- `test_image.py` - Command-line tool for testing single images
- `app.py` - Flask web application for online predictions
- `data/` - Training data directory with chocolate images
- `templates/` - HTML templates for the web interface

## Usage

### Training the Model
```bash
python train.py
```
This will train the model and save it as `model.pkl`

### Testing an Image
```bash
python test_image.py path/to/your/chocolate/image.jpg
```

### Running the Web App
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

## Model Performance
- Training Accuracy: 100%
- Validation Accuracy: 96.43%

## Technologies Used
- Python
- scikit-learn
- NumPy
- Pillow (PIL)
- Flask
