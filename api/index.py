from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# --------------------------------------------------
# 1️⃣ Load TFLite Model
# --------------------------------------------------
MODEL_PATH = "enhanced_model_web_final.tflite"

CLASS_NAMES = [
    "cherry_diseased","cherry_healthy",
    "grape_diseased","grape_healthy",
    "peach_diseased","peach_healthy",
    "pepper_bell_diseased","pepper_bell_healthy",
    "potato_diseased","potato_healthy",
    "tomato_diseased","tomato_healthy"
]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Model loaded successfully!")
print("Input:", input_details)
print("Output:", output_details)

# --------------------------------------------------
# 2️⃣ Preprocess (EfficientNet-style)
# --------------------------------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)  # EfficientNet preprocessing
    arr = np.expand_dims(arr, axis=0)
    return arr

# --------------------------------------------------
# 3️⃣ Auto Temperature Softmax Sharpening
# --------------------------------------------------
def smart_softmax(logits):
    """
    Automatically adjusts softmax 'temperature' based on entropy.
    """
    logits = np.array(logits, dtype=np.float32)

    # Compute base softmax
    base_probs = tf.nn.softmax(logits).numpy()
    entropy = -np.sum(base_probs * np.log(base_probs + 1e-8))

    # Adjust temperature: higher entropy (flatter) => lower temperature
    temperature = np.clip(entropy / 2.5, 0.25, 1.0)
    adjusted = logits / temperature

    probs = tf.nn.softmax(adjusted).numpy()
    return probs, temperature

# --------------------------------------------------
# 4️⃣ Predict Function
# --------------------------------------------------
def predict_image(image_bytes):
    input_data = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    probs, temperature = smart_softmax(output_data)

    max_idx = int(np.argmax(probs))
    confidence = float(probs[max_idx]) * 100
    label = CLASS_NAMES[max_idx]

    print(f"🌿 Predicted: {label} ({confidence:.2f}%) | AutoTemp={temperature:.2f}")
    return label, (temperature*100)

# --------------------------------------------------
# 5️⃣ Flask Routes
# --------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_bytes = file.read()
    label, confidence = predict_image(image_bytes)
    return jsonify({"label": label, "confidence": f"{confidence:.2f}"})

# --------------------------------------------------
# 6️⃣ Run the app
# --------------------------------------------------
if __name__ == '__main__':
    app.run(port=5000,debug=True)
