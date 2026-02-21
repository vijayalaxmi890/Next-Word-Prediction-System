import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ------------------ Models ------------------

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, bidirectional=False):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ------------------ Accuracy & Evaluation ------------------

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        loss = nn.CrossEntropyLoss()(outputs, y_test)
        acc = calculate_accuracy(outputs, y_test)
    return acc, loss.item()

# ------------------ Load Model ------------------

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    config = checkpoint['config']
    model_type = config['model_type']

    if model_type == "rnn":
        model = TextRNN(config['vocab_size'], config['embed_size'], config['hidden_size'])
    elif model_type == "lstm":
        model = TextLSTM(config['vocab_size'], config['embed_size'], config['hidden_size'], bidirectional=False)
    elif model_type == "bilstm":
        model = TextLSTM(config['vocab_size'], config['embed_size'], config['hidden_size'], bidirectional=True)
    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

# ------------------ Prediction Functions ------------------

def predict_rnn(user_input, tokenizer, model_path, top_k=5):
    model = load_model(model_path)
    tokens = [tokenizer.get(w, tokenizer['<unk>']) for w in user_input.strip().split()]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)
        top_indices = torch.topk(probs, top_k).indices.squeeze().tolist()

    id_to_word = {v: k for k, v in tokenizer.items()}
    return [id_to_word.get(idx, "<unk>") for idx in top_indices]

def predict_lstm(user_input, tokenizer, model_path, top_k=5):
    return predict_rnn(user_input, tokenizer, model_path, top_k)

def predict_gpt2(input_text, model_path='gpt2', top_k=5):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_length = input_ids.shape[1]

    outputs = model.generate(
        input_ids,
        max_length=input_length + 5,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=top_k,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    seen = set()
    predictions = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True).strip()
        continuation = decoded[len(input_text):].strip()
        next_word = continuation.split()[0] if continuation else "<unk>"

        if next_word not in seen:
            seen.add(next_word)
            predictions.append(next_word)

        if len(predictions) >= top_k:
            break

    return predictions
