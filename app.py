import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the sentence transformer model
model_sentence = SentenceTransformer('all-MiniLM-L6-v2')

# Recreate the model architecture (same as in your Colab)
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

# Create the model
nn_model = create_model()

# Try loading weights instead of the full model
try:
    nn_model.load_weights('phq9_model.keras')
    print("Successfully loaded model weights")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Starting with untrained model. Please ensure model weights are available.")

def pred(text):
    # Generate the sentence embedding
    embedding = model_sentence.encode([text])
    
    # Get the predictions (probabilities between 0 and 1)
    prediction = nn_model.predict(embedding)
    
    # Convert probabilities to binary labels (threshold 0.35)
    pred_binary = (prediction > 0.35).astype(int)
    
    # Calculate percentage of "yes" answers
    percent = (np.sum(pred_binary) / 9) * 100
    return f"{int(percent)}% depressed {pred_binary}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Get prediction
    result = pred(text)
    
    # Extract the percentage (first two characters) from the result
    percentage_str = result.split('%')[0]
    percentage = int(percentage_str)
    
    return jsonify({'percentage': percentage, 'full_result': result})

if __name__ == '__main__':
    app.run(debug=True)