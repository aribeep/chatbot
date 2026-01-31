import torch
import torch.nn as nn
import pandas as pd
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GRUIntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, bidirectional=True):
        super(GRUIntentClassifier, self).__init__()
        # Initialize an Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize a GRU layer
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Initialize a Linear layer
        # If bidirectional, the output hidden size is hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, input_ids):
        # Pass the input_ids through the Embedding layer
        embedded = self.embedding(input_ids)

        # Pass the output of the embedding layer through the GRU layer
        # GRU outputs: output (seq_len, batch, num_directions * hidden_size)
        # hidden (num_layers * num_directions, batch, hidden_size)
        gru_output, hidden = self.gru(embedded)

        # Process GRU output to get a fixed-size representation for classification.
        # hidden: (num_layers * num_directions, batch, hidden_size)

        if self.gru.bidirectional:
            # hidden[-2, :, :] is the last forward hidden state
            # hidden[-1, :, :] is the last backward hidden state
            final_hidden_state = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # For unidirectional GRU, the last hidden state is hidden[-1, :, :]
            final_hidden_state = hidden[-1, :, :]

        # Pass this representation through the Linear layer
        logits = self.fc(final_hidden_state)

        return logits
    

# Path to the downloaded model file on your local machine
PATH = "./CVNL_INTENT.pt"

# Instantiate the model
model = GRUIntentClassifier(30522,100, 256, 2, 26)

# Load the state_dict, mapping to CPU if necessary (common when moving from Colab GPU)
# Use weights_only=True for security and best practice
xx = model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'), weights_only=True))

# Set the model to evaluation mode for inference
model.eval()

with open('id_to_label.json', 'r') as file:
        id_to_label = json.load(file)

from transformers import AutoTokenizer

# Initialize a suitable tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def predict_intent(sentence):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        # Tokenize the input sentence
        inputs = tokenizer(sentence, truncation=True, padding='max_length', max_length=200, return_tensors='pt')

        # Move inputs to the correct device
        input_ids = inputs['input_ids'].to(device)

        # Get model predictions
        outputs = model(input_ids)

        # Get the predicted class ID
        predicted_id = torch.argmax(outputs, dim=1).item()

        # Convert ID back to label
        predicted_label = id_to_label[str(predicted_id)]

        return predicted_label