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

nn_model = create_model()

try:
    nn_model.load_weights('phq9_model.keras')
except Exception as e:
    print(f"Failed to load model weights: {e}")

HF_API_KEY = os.getenv("HF_API_KEY")

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(
        "https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2",
        headers=headers,
        json={"inputs": text}
    )
    embedding = response.json().get("embedding")
    return np.array([embedding])

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
    text = request.form['text']
    result = pred(text)
    percentage_str = result.split('%')[0]
    percentage = int(percentage_str)
    return jsonify({'percentage': percentage, 'full_result': result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

