// src/PredictionForm.js
import React, { useState } from 'react';
import './PredictionForm.css';

const PredictionForm = ({ onPredictionSubmit }) => {
  const [inputData, setInputData] = useState({
    // Add input fields corresponding to your dataset attributes
    age: '',
    cigsPerDay: '',
    totChol: '',
    sysBP: '',
    diaBP: '',
    heartRate: '',
    glucose: '',
    male: '',
    education: '',
    BPMeds: '',
    prevalentStroke: '',
    prevalentHyp: '',
    diabetes: '',

    // Add more attributes as needed
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    const numericalValue = parseFloat(value)
    setInputData({ ...inputData, [name]: numericalValue });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredictionSubmit(inputData);
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <h2>Can I check your Heart</h2>
      {/* Input fields for each attribute */}
      <label>
      age:
        <input type="text" name="age" value={inputData.attribute1} onChange={handleChange} />
      </label>
      <br />
      <label>
      cigsPerDay:
        <input type="text" name="cigsPerDay" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
      totChol:
        <input type="text" name="totChol" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
       sysBP:
        <input type="text" name="sysBP" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
      diaBP:
        <input type="text" name="diaBP" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
      heartRate:
        <input type="text" name="heartRate" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
       male:
        <input type="text" name="male" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
      education:
        <input type="text" name="education" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
       BPMeds:
        <input type="text" name="BPMeds" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
      prevalentStroke:
        <input type="text" name="prevalentStroke" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
      prevalentHyp:
        <input type="text" name="prevalentHyp" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />
      <label>
      diabetes:
        <input type="text" name="diabetes" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />

      <label>
      glucose:
        <input type="text" name="glucose" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />

      <label>
        BMI:
        <input type="text" name="BMI" value={inputData.attribute2} onChange={handleChange} />
      </label>
      {/* Add more input fields as needed */}
      <br />

      <button type="submit">Predict</button>
    </form>
  );
};

export default PredictionForm;


