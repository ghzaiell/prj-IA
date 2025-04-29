import numpy as np
from PIL import Image
import pickle
import sys

def load_and_preprocess_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    # Flatten the image for RandomForest
    img_flat = img_array.reshape(1, -1)
    return img_flat

def main():
    if len(sys.argv) < 2:
        print("Please provide an image path")
        print("Usage: python test_image.py <path_to_image>")
        return

    image_path = sys.argv[1]
    
    try:
        # Load the model and class names
        print("Loading model...")
        with open('model.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        model = model_info['model']
        class_names = model_info['class_names']
        
        # Preprocess the image
        print("Processing image...")
        img_array = load_and_preprocess_image(image_path)
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_array)
        probabilities = model.predict_proba(img_array)
        
        # Get the predicted class and probability
        predicted_class = class_names[prediction[0]]
        probability = probabilities[0][prediction[0]] * 100
        
        print("\nResults:")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {probability:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()