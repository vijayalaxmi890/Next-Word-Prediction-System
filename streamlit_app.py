import streamlit as st
import pickle
from utils import predict_rnn, predict_lstm, predict_gpt2

# =========================
# LOAD TOKENIZER
# =========================
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Next Word Prediction", layout="centered")
st.title("🧠 Next Word Prediction")

# =========================
# SESSION STATE INIT
# =========================
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "predictions" not in st.session_state:
    st.session_state.predictions = []

# =========================
# TEXT INPUT
# =========================
st.session_state.input_text = st.text_input(
    "Enter text",
    value=st.session_state.input_text,
    placeholder="e.g. machine learning models",
)

# =========================
# MODEL SELECTION
# =========================
model_choice = st.selectbox(
    "Choose a model",
    ["RNN", "LSTM", "Bi-LSTM", "GPT-2"],
)

# =========================
# ACTION BUTTONS
# =========================
col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("Predict Next")

with col2:
    clear_clicked = st.button("Clear")

if clear_clicked:
    st.session_state.input_text = ""
    st.session_state.predictions = []
    st.rerun()

# =========================
# PREDICTION LOGIC
# =========================
if predict_clicked and st.session_state.input_text.strip():

    text = st.session_state.input_text.strip()

    if model_choice == "RNN":
        preds = predict_rnn(text, tokenizer, "models/rnn_model.pt", top_k=5)

    elif model_choice == "LSTM":
        preds = predict_lstm(text, tokenizer, "models/lstm_model.pt", top_k=5)

    elif model_choice == "Bi-LSTM":
        preds = predict_lstm(text, tokenizer, "models/bilstm_model.pt", top_k=5)

    else:  # GPT-2
        preds = predict_gpt2(text, model_path="gpt2", top_k=5)

    st.session_state.predictions = preds

# =========================
# SHOW PREDICTIONS
# =========================
if st.session_state.predictions:
    st.subheader("Predicted next words")

    for i, word in enumerate(st.session_state.predictions):
        if st.button(word, key=f"pred_{i}"):
            st.session_state.input_text += " " + word
            st.session_state.predictions = []
            st.rerun()
