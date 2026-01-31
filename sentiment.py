import torch
import torch.nn as nn
import pandas as pd
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size=13000, embed_dim=100, hidden_dim=64, num_classes=3, pad_idx=0, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, h_n = self.gru(emb)            # h_n: (1, batch, hidden_dim)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits
    

# Path to the downloaded model file on your local machine
PATH = "./CVNL_SENTIMENT.pt"

# Instantiate the model
model = GRUClassifier()

# Load the state_dict, mapping to CPU if necessary (common when moving from Colab GPU)
# Use weights_only=True for security and best practice
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'), weights_only=True))

# Set the model to evaluation mode for inference
model.eval()

def predict_sentiment(text, word2idx, max_len=50):

    tokens = text.split()
    seq = [word2idx.get(w, 1) for w in tokens]  # UNK=1
    seq = seq[:max_len] + [0] * (max_len - len(seq))

    x = torch.tensor([seq], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    sentiment_labels = ["negative", "neutral", "positive"]
    return sentiment_labels[pred]