import time
import psutil
import pickle
import pandas as pd
from utils import predict_rnn, predict_lstm, predict_gpt2

# =========================
# LOAD TOKENIZER
# =========================
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

process = psutil.Process()

# =========================
# TEST SENTENCES
# =========================
test_sentences = [
    "machine learning models are",
    "deep learning techniques can",
    "artificial intelligence is used for",
    "natural language processing enables",
    "neural networks are capable of"
]

# =========================
# EVALUATION FUNCTIONS
# =========================
def evaluate_dl_model(model_name, predict_fn, model_path):
    latencies = []
    cpu_usages = []
    memory_usages = []

    for sentence in test_sentences:
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        predict_fn(sentence, tokenizer, model_path, top_k=5)

        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 * 1024)

        latencies.append((end_time - start_time) * 1000)
        memory_usages.append(end_mem - start_mem)

    return {
        "Model": model_name,
        "Avg Latency (ms)": round(sum(latencies) / len(latencies), 2),
        "Avg Memory Usage (MB)": round(sum(memory_usages) / len(memory_usages), 2),
    }

def evaluate_gpt2():
    latencies = []
    memory_usages = []

    for sentence in test_sentences:
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        predict_gpt2(sentence, top_k=5)

        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 * 1024)

        latencies.append((end_time - start_time) * 1000)
        memory_usages.append(end_mem - start_mem)

    return {
        "Model": "GPT-2",
        "Avg Latency (ms)": round(sum(latencies) / len(latencies), 2),
        "Avg Memory Usage (MB)": round(sum(memory_usages) / len(memory_usages), 2),
    }

# =========================
# RUN EVALUATION
# =========================
results = []

results.append(evaluate_dl_model(
    "RNN",
    predict_rnn,
    "models/rnn_model.pt"
))

results.append(evaluate_dl_model(
    "LSTM",
    predict_lstm,
    "models/lstm_model.pt"
))

results.append(evaluate_dl_model(
    "Bi-LSTM",
    predict_lstm,
    "models/bilstm_model.pt"
))

results.append(evaluate_gpt2())

# =========================
# SAVE RESULTS
# =========================
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\nEvaluation completed successfully!\n")
print(df)
