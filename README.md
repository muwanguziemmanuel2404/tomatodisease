# Tomato Disease Detection System

## Overview

Tomato Disease Detection System is a web-based machine learning application that uses deep learning to identify diseases affecting tomato plants from leaf images. Users can upload an image of a tomato leaf through an intuitive web interface and receive an instant prediction of the detected disease.

The project combines a React frontend, a Flask backend API, and a PyTorch deep learning model to deliver accurate and efficient disease classification.

---

## Features

* Upload tomato leaf images for analysis
* Real-time disease prediction
* Deep learning-based image classification
* RESTful API architecture
* Responsive React user interface
* Flask backend for model inference
* PyTorch model integration

---

## Technology Stack

### Frontend

* React.js
* HTML5
* CSS3
* JavaScript

### Backend

* Flask
* Flask-CORS

### Machine Learning

* PyTorch
* TorchVision
* NumPy
* Pillow

---

## Project Architecture

```text
Tomato-Disease-Detection/
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── app.py
│   ├── model.pth
│   ├── requirements.txt
│   └── utils/
│
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Tomato-Disease-Detection
```

### Backend Setup

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

python app.py
```

Backend will run on:

```text
http://localhost:5000
```

### Frontend Setup

```bash
cd frontend

npm install

npm start
```

Frontend will run on:

```text
http://localhost:3000
```

---

## How to Use

1. Open the frontend application.
2. Upload a tomato leaf image.
3. Submit the image for analysis.
4. The Flask API processes the image using the trained PyTorch model.
5. View the predicted disease class and confidence score.

---

## Model Information

The application uses a trained PyTorch model stored as a `.pth` file. The model performs image classification on tomato leaf images and predicts the corresponding disease category.

---

## Deployment

### Frontend

Deployed using Vercel.

### Backend

Deployed using Render.

---

## Future Enhancements

* Disease treatment recommendations
* Severity estimation
* Mobile application support
* Multi-crop disease detection
* User authentication and prediction history

---

## Author

Emmanuel Muwanguzi

---

