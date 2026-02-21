import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from utils import TextRNN, TextLSTM, calculate_accuracy, evaluate_model
from collections import Counter
from sklearn.model_selection import train_test_split
import os

# =========================
# CONFIG (OPTIMIZED FOR i3 + 8GB RAM)
# =========================
DATA_PATH = "data.txt"
EMBED_SIZE = 32
HIDDEN_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 10
MIN_WORD_FREQ = 2

DEVICE = torch.device("cpu")

# =========================
# LOAD DATA
# =========================
def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f if line.strip()]
    return lines

corpus = load_corpus(DATA_PATH)

# =========================
# TOKENIZER
# =========================
def build_tokenizer(corpus, min_freq=1):
    words = []
    for sentence in corpus:
        words.extend(sentence.split())

    vocab = [
        word for word, freq in Counter(words).items()
        if freq >= min_freq
    ]

    vocab = ["<pad>", "<unk>"] + sorted(vocab)
    return {word: idx for idx, word in enumerate(vocab)}

tokenizer = build_tokenizer(corpus, MIN_WORD_FREQ)
vocab_size = len(tokenizer)

# Save tokenizer
os.makedirs("models", exist_ok=True)
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# =========================
# DATA PREPARATION
# =========================
def prepare_data(corpus, tokenizer):
    X, y = [], []

    for sentence in corpus:
        words = sentence.split()
        for i in range(1, min(len(words), MAX_SEQ_LEN)):
            input_seq = words[:i]
            target = words[i]

            input_ids = [tokenizer.get(w, tokenizer["<unk>"]) for w in input_seq]
            target_id = tokenizer.get(target, tokenizer["<unk>"])

            X.append(input_ids)
            y.append(target_id)

    return X, y

X_raw, y_raw = prepare_data(corpus, tokenizer)

# Padding
max_len = max(len(seq) for seq in X_raw)
X_padded = [
    seq + [tokenizer["<pad>"]] * (max_len - len(seq))
    for seq in X_raw
]

X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y_raw, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, model_name, config):
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🔹 Training {model_name.upper()} model")

    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(outputs, y_train)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{model_name}] "
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {acc:.4f}"
            )

    # Save model
    torch.save(
        {
            "model": model.state_dict(),
            "config": config
        },
        f"models/{model_name}_model.pt"
    )

    test_acc, test_loss = evaluate_model(model, X_test, y_test)
    print(
        f"[{model_name}] Test Accuracy: {test_acc:.4f} | "
        f"Test Loss: {test_loss:.4f}"
    )

# =========================
# TRAIN MODELS
# =========================

# RNN
rnn_model = TextRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
train_model(
    rnn_model,
    "rnn",
    {
        "model_type": "rnn",
        "vocab_size": vocab_size,
        "embed_size": EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE,
    },
)

# LSTM
lstm_model = TextLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, bidirectional=False)
train_model(
    lstm_model,
    "lstm",
    {
        "model_type": "lstm",
        "vocab_size": vocab_size,
        "embed_size": EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE,
    },
)

# Bi-LSTM
bilstm_model = TextLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, bidirectional=True)
train_model(
    bilstm_model,
    "bilstm",
    {
        "model_type": "bilstm",
        "vocab_size": vocab_size,
        "embed_size": EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE,
    },
)
