import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model architecture (MUST match Colab exactly) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_p=0.2):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),     # 256 -> 128
            block(32, 64),    # 128 -> 64
            block(64, 128),   # 64 -> 32
            block(128, 256),  # 32 -> 16
            block(256, 512),  # 16 -> 8
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


# --- Loading checkpoint + building transforms ---
def load_aircraft_model(checkpoint_path: str = "./aircraft_family_best.pt"):
    """
    Loads the 'integration-ready' checkpoint saved from Colab.
    Returns: (model, idx_to_label, preprocess_transform)
    """
    # NOTE: This checkpoint is a dict (not just a state_dict)
    ckpt = torch.load(checkpoint_path, map_location="cpu")  # keep on CPU first

    idx_to_label = ckpt["idx_to_label"]
    img_size = ckpt.get("img_size", 256)
    dropout_p = ckpt.get("dropout_p", 0.2)
    num_classes = ckpt.get("num_classes", len(idx_to_label))

    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])

    model = SimpleCNN(num_classes=num_classes, dropout_p=dropout_p)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, idx_to_label, preprocess


# Cache so the app doesn't reload every request
_MODEL = None
_IDX_TO_LABEL = None
_PREPROCESS = None

def _ensure_loaded():
    global _MODEL, _IDX_TO_LABEL, _PREPROCESS
    if _MODEL is None:
        _MODEL, _IDX_TO_LABEL, _PREPROCESS = load_aircraft_model("./aircraft_family_best.pt")


def predict_aircraft(image_path: str):
    """
    Predict aircraft family label from an image file path.
    Returns: (predicted_label, confidence_float_0_to_1)
    """
    _ensure_loaded()

    img = Image.open(image_path).convert("RGB")

    x = _PREPROCESS(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    with torch.no_grad():
        logits = _MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        conf = float(probs[pred_idx].item())

    # In your Colab, idx_to_label keys are ints -> labels
    label = _IDX_TO_LABEL[pred_idx] if isinstance(_IDX_TO_LABEL, dict) else str(pred_idx)
    return label, conf
