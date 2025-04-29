from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import pickle
import io
import base64

app = Flask(__name__)

def load_and_preprocess_image(image_file):
    # Read and preprocess the image
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    # Flatten the image for RandomForest
    img_flat = img_array.reshape(1, -1)
    return img_flat

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            # Get the image from the POST request
            image_file = request.files['image']
            
            # Load the model
            with open('model.pkl', 'rb') as f:
                model_info = pickle.load(f)
            model = model_info['model']
            class_names = model_info['class_names']
            
            # Preprocess the image
            img_array = load_and_preprocess_image(image_file)
            
            # Make prediction
            prediction = model.predict(img_array)
            probabilities = model.predict_proba(img_array)
            
            # Get the predicted class and probability
            predicted_class = class_names[prediction[0]]
            probability = probabilities[0][prediction[0]] * 100
            
            result = {
                'class': predicted_class,
                'confidence': f'{probability:.2f}%'
            }
            
        except Exception as e:
            result = {'error': str(e)}
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
