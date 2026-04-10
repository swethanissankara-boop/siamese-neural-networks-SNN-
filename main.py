from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import layers, models
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io
import json

# 1. Initialize the app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. BUILD THE ARCHITECTURE LOCALLY (Bypasses the version crash!)
print("Building Siamese Architecture...")
def build_twin_network(input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inputs, x)

input_a = layers.Input(shape=(28, 28, 1))
input_b = layers.Input(shape=(28, 28, 1))
twin_network = build_twin_network()

output_a = twin_network(input_a)
output_b = twin_network(input_b)
difference = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([output_a, output_b])
output = layers.Dense(1, activation='sigmoid')(difference)

# This is your model shell, ready to receive the learned brain
model = models.Model(inputs=[input_a, input_b], outputs=output)

# 3. LOAD ONLY THE WEIGHTS
print("Loading Weights...")
model.load_weights('fashion_siamese.keras')
print("Model Ready!")

# 4. Image Preprocessing for the AI
def preprocess_for_fashion_mnist(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=(0, -1))

# --- ENDPOINT 1: THE UPLOAD TESTER ---
@app.post("/compare")
async def compare_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    bytes1 = await image1.read()
    bytes2 = await image2.read()

    tensor1 = preprocess_for_fashion_mnist(bytes1)
    tensor2 = preprocess_for_fashion_mnist(bytes2)

    distance = model.predict([tensor1, tensor2])[0][0]
    
    threshold = 0.5
    is_match = bool(distance < threshold)

    return {
        "distance": float(distance),
        "is_match": is_match,
        "message": "These items look similar!" if is_match else "These items look different."
    }

# --- ENDPOINT 2: THE DASHBOARD DATA ---
@app.get("/dashboard-data")
async def get_dashboard_data():
    limit = 1000 # Limit points so the browser doesn't crash
    
    coords = np.load('data/embeddings_2d.npy')[:limit]
    labels = np.load('data/y_test.npy')[:limit]
    
    with open('data/images_base64.json', 'r') as f:
        images = json.load(f)[:limit]
        
    class_names = {
        0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
    }
    
    text_labels = [class_names[int(lbl)] for lbl in labels]

    return {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "labels": text_labels,
        "images": images
    }