import numpy as np
from PIL import Image
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set up parameters
IMG_SIZE = 224
BATCH_SIZE = 32

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                # Flatten the image for RandomForest
                img_flat = img_array.reshape(-1)
                images.append(img_flat)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

def main():
    # Load and preprocess data
    data_dir = "data"
    print("Loading and preprocessing data...")
    X, y, class_names = load_and_preprocess_data(data_dir)
    
    # Split data into train and validation sets
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Save the model and class names
    print("Saving the model...")
    model_info = {
        'model': model,
        'class_names': class_names
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("Model trained and saved successfully!")
    print(f"Classes: {class_names}")

if __name__ == "__main__":
    main()