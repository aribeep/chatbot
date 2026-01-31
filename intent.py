import torch
models = torch.load('CVNL_INTENT.pth')

device='cuda'
def predict_intent(sentence):
    with torch.no_grad():
        # Tokenize the input sentence
        inputs = models.tokenizer(sentence, truncation=True, padding='max_length', max_length=models.max_uniform_length, return_tensors='pt')

        # Move inputs to the correct device
        input_ids = inputs['input_ids'].to(device)

        # Get model predictions
        outputs = models(input_ids)

        # Get the predicted class ID
        predicted_id = torch.argmax(outputs, dim=1).item()

        # Convert ID back to label
        predicted_label = models.id_to_label[predicted_id]

        return predicted_label