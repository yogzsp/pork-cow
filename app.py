import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_PATH = "meat_vs_pork_transfer.keras"  # Replace with your model's path
model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an image is part of the request
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Read the image file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))  # Resize to model input size

        # Preprocess the image
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Prediction
        prediction = model.predict(img_array)

        # Interpret the prediction
        predicted_class = np.argmax(prediction, axis=1)
        labels = ["Daging Sapi", "Daging Babi"]  # Adjust labels according to your dataset

        return jsonify({"prediction": labels[predicted_class[0]]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# CORS Handling (Optional)
from flask_cors import CORS
CORS(app)  # This will allow all origins by default

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
