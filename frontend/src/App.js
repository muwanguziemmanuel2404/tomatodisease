import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL =
  process.env.REACT_APP_API_URL || "http://localhost:5000";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);

  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");

  const [recommendation, setRecommendation] = useState("");

  const [gradcam, setGradcam] = useState("");

  const [loading, setLoading] = useState(false);

  const handleImage = (e) => {
    const file = e.target.files[0];

    setImage(file);

    setPreview(URL.createObjectURL(file));
  };

  const handleSubmit = async () => {
    if (!image) return;

    const formData = new FormData();

    formData.append("file", image);

    setLoading(true);

    try {
      const response = await axios.post(
        `${API_URL}/predict`,
        formData
      );

      setPrediction(response.data.prediction);

      setConfidence(response.data.confidence);

      setRecommendation(response.data.recommendation);

      setGradcam(response.data.gradcam);

    } catch (error) {
      console.error(error);
      alert("Prediction failed");
    }

    setLoading(false);
  };

  return (
    <div className="container">

      <h1>🍅 Tomato Disease Detection AI</h1>

      <p className="subtitle">
        Upload a tomato leaf image for AI diagnosis
      </p>

      <div className="upload-box">

        <input
          type="file"
          accept="image/*"
          onChange={handleImage}
        />

        {preview && (
          <img
            src={preview}
            alt="preview"
            className="preview"
          />
        )}

        <button onClick={handleSubmit}>
          Predict Disease
        </button>

      </div>

      {loading && <h2>Analyzing image...</h2>}

      {prediction && (
        <div className="result-card">

          <h2>Prediction Result</h2>

          <h3>{prediction}</h3>

          <p>
            Confidence: <strong>{confidence}%</strong>
          </p>

          {gradcam && (
            <>
              <h3>AI Explainability</h3>

              <img
                src={`data:image/png;base64,${gradcam}`}
                alt="GradCAM"
                className="gradcam"
              />
            </>
          )}

          <div className="recommendation">

            <h3>AI Farming Recommendation</h3>

            <p>{recommendation}</p>

          </div>

        </div>
      )}

    </div>
  );
}

export default App;
