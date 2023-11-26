// src/Result.js
import React, { useState, useEffect } from 'react';
import PredictionForm from './PredictionForm';
import './Result.css';

const Result = () => {
  const [auc, setAuc] = useState(null);
  const [classificationRep, setClassificationRep] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  

  useEffect(() => {
    // Fetch data from Flask API endpoint
    fetch('/api/results')
      .then(response => response.json())
      .then(data => {
        setAuc(data.auc);
        setClassificationRep(data.classification_rep);
      })
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  const handlePredictionSubmit = (inputData) => {
    // Make a call to the Flask API endpoint for making predictions
    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(inputData),
    })
      .then(response => response.json())
      .then(data => {
        console.log('Prediction Response:', data);
        setPredictionResult(data);
      })
      .catch(error => console.error('Error making prediction:', error));
  };


  return (
    <div className="result-container">
      <h1>Framingham Heart Study Analysis</h1>
      <h2>Model AUC: {auc}</h2>
      <h2>Classification Report:</h2>
      {classificationRep && (
        <table className="classification-table" border="1">
          <thead>
            <tr>
              <th></th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0</td>
              <td>{classificationRep['0']['precision']}</td>
              <td>{classificationRep['0']['recall']}</td>
              <td>{classificationRep['0']['f1-score']}</td>
              <td>{classificationRep['0']['support']}</td>
            </tr>
            <tr>
              <td>1</td>
              <td>{classificationRep['1']['precision']}</td>
              <td>{classificationRep['1']['recall']}</td>
              <td>{classificationRep['1']['f1-score']}</td>
              <td>{classificationRep['1']['support']}</td>
            </tr>
          </tbody>
        </table>
      )}

    <PredictionForm onPredictionSubmit={handlePredictionSubmit} />
    {predictionResult && (
        <div className="prediction-result">
          <h2>Prediction Result</h2>
          <p>Predicted Class: {predictionResult.predicted_class}</p>
          <p>Confidence: {predictionResult.confidence}</p>
        </div>
      )}
    </div>
  );
};

export default Result;
