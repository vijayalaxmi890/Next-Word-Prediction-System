# Next Word Prediction System

## Overview
This project implements a **Next Word Prediction system** using advanced deep learning models such as **RNN, LSTM, BiLSTM**, and **GPT-2**. The system predicts the most probable next word based on the input text, helping improve typing efficiency, chatbots, and language modeling applications.

---

## Features
- Predicts the next word in a given sentence.
- Supports multiple deep learning models:
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
  - Bidirectional LSTM (BiLSTM)
  - GPT-2 (pre-trained transformer)
- Streamlit web application for real-time predictions.
- Easy to extend and train on custom datasets.

---

## Project Structure


NWP/
│
├── models/ # Trained model files
├── data # Dataset for training
├── train.py # Script to train models
├── evaluate_models.py # Script to evaluate model performance
├── streamlit_app.py # Web app for interactive predictions
├── utils.py # Utility functions
├── requirements.txt # Project dependencies
└── README.md # Project documentation


> **Note:** `venv/` and `__pycache__/` are excluded from GitHub via `.gitignore`.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/vijayalaxmi890/Next-Word-Prediction-System.git
cd Next-Word-Prediction-System

Create a virtual environment:

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

Install dependencies:

pip install -r requirements.txt
Usage
Train the Model
python train.py
Evaluate Model Performance
python evaluate_models.py
Run Streamlit Web App
streamlit run streamlit_app.py

Open the browser link displayed in the terminal to interact with the app.

#Requirements

Python 3.8+

TensorFlow / PyTorch

Streamlit

NumPy, Pandas, Matplotlib

All dependencies are listed in requirements.txt.

#How It Works

Preprocessing: Text data is cleaned, tokenized, and converted to sequences.

Model Training: Sequences are fed into RNN, LSTM, or BiLSTM models to learn word patterns.

Prediction: Given an input sequence, the model predicts the next word based on learned probabilities.

Web App: Streamlit interface allows users to input text and get real-time predictions.

#Sample Output

Input: "I love to"
Predicted Next Words: "eat", "play", "code"

#Future Enhancements

Integrate GPT-3 or GPT-4 for more accurate predictions.

Add support for multi-language text prediction.

Deploy as a web service using Flask or FastAPI.

Author - Vijayalaxmi Biradar
