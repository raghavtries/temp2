import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import requests

app = Flask(__name__)

def create_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Dense(128, activation='relu', input_shape=(384,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(9, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load model
nn_model = create_model()
try:
    nn_model.load_weights('phq9_model.keras')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"[Failed] failed to load model weights: {e}")

HF_API_KEY = os.getenv("HF_API_KEY")

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
        headers=headers,
        json={"inputs": text}
    )

    print("📡 HuggingFace API response:", response.text)

    embedding = response.json()
    if not isinstance(embedding, list) or not embedding or not isinstance(embedding[0], list):
        raise ValueError("[failed] invalid embedding received from HuggingFace API.")

    return np.array([embedding[0]])

def pred(text):
    embedding = get_embedding(text)
    prediction = nn_model.predict(embedding)
    pred_binary = (prediction > 0.35).astype(int)
    percent = (np.sum(pred_binary) / 9) * 100
    return f"{int(percent)}% depressed {pred_binary}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        print(f"Received input: {text}")
        result = pred(text)
        print(f"Prediction result: {result}")
        percentage_str = result.split('%')[0]
        percentage = int(percentage_str)
        return jsonify({'percentage': percentage, 'full_result': result})
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
