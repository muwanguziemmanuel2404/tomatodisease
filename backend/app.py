#```python
from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import traceback
import base64
import cv2
import numpy as np

from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms

# ==========================================================
# FLASK SETUP
# ==========================================================

app = Flask(__name__)
CORS(app)

# ==========================================================
# RTL-NET MODEL
# ==========================================================

class RTLNet(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = "models/RTLNet_EarlyBlight.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

DEVICE = torch.device("cpu")

model = RTLNet(num_classes=2)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully")

# ==========================================================
# CLASS LABELS
# ==========================================================

CLASS_NAMES = [
    "Tomato Early Blight",
    "Tomato Healthy"
]

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor

# ==========================================================
# PREDICT
# ==========================================================

def predict_disease(image):

    input_tensor = preprocess_image(image)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_idx = int(
            torch.argmax(probabilities, dim=1)
        )

        confidence = float(
            probabilities[0][predicted_idx]
        ) * 100

    prediction = CLASS_NAMES[predicted_idx]

    if prediction == "Tomato Early Blight":

        recommendation = (
            "Early blight detected. "
            "Remove infected leaves, improve airflow, "
            "avoid overhead watering and apply a suitable fungicide."
        )

    else:

        recommendation = (
            "Leaf appears healthy. "
            "Continue normal irrigation, fertilization "
            "and disease monitoring."
        )

    return prediction, confidence, recommendation

# ==========================================================
# SIMPLE HEATMAP
# ==========================================================

def create_heatmap(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        (300, 300)
    )

    heatmap = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET
    )

    success, buffer = cv2.imencode(
        ".png",
        heatmap
    )

    if not success:
        return ""

    return base64.b64encode(
        buffer
    ).decode("utf-8")

# ==========================================================
# API ROUTE
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        image = Image.open(
            file
        ).convert("RGB")

        prediction, confidence, recommendation = (
            predict_disease(image)
        )

        heatmap = create_heatmap(image)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "recommendation": recommendation,
            "gradcam": heatmap
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "running",
        "model": "RTLNet Early Blight"
    })

# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
#```