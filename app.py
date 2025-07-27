import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import json
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_PATH = "meat_vs_pork_transfer.keras"
model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # === Ekstraksi Gambar ===
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_size = img.size
        img = img.resize((224, 224))

        # === Preprocessing ===
        img_array = np.array(img) / 255.0
        normalized_preview = img_array[:2, :2, :].tolist()  # hanya potongan kecil

        img_array = np.expand_dims(img_array, axis=0)

        # === Prediksi ===
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        labels = ["Daging Sapi", "Daging Babi"]
        confidence = float(np.max(prediction))

        # === Akurasi Model ===
        try:
            with open("training_info.json") as f:
                training_info = json.load(f)
        except:
            training_info = {"val_accuracy": None, "epochs": None}

        return jsonify({
            "ekstraksi": {
                "original_size": original_size,
                "resized_to": [224, 224]
            },
            "preprocessing": {
                "normalized_sample": normalized_preview
            },
            "konversi": {
                "predicted_label": labels[predicted_class[0]],
                "confidence": confidence
            },
            "model_info": training_info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Enable CORS
from flask_cors import CORS
CORS(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
