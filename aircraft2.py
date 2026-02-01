# aircraft2.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Label mapping from FGVC variants.txt
# -----------------------------
def build_idx_to_label_from_variants_txt(variants_txt_path: str):
    with open(variants_txt_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return {i: lbl for i, lbl in enumerate(labels)}


# -----------------------------
# Model definition (must match training)
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // r)
        self.fc2 = nn.Linear(channels // r, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, r=16)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class AircraftVariantCNN_v5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.stage1 = nn.Sequential(ResidualSEBlock(64, 64), ResidualSEBlock(64, 64))
        self.stage2 = nn.Sequential(ResidualSEBlock(64, 128, stride=2), ResidualSEBlock(128, 128))
        self.stage3 = nn.Sequential(ResidualSEBlock(128, 256, stride=2), ResidualSEBlock(256, 256))
        self.stage4 = nn.Sequential(ResidualSEBlock(256, 512, stride=2), ResidualSEBlock(512, 512))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x)


# -----------------------------
# Cache model so we don't reload each request
# -----------------------------
_VARIANT_MODEL = None
_VARIANT_IDX_TO_LABEL = None
_VARIANT_PREPROCESS = None


def load_variant_model_state_dict(
    checkpoint_path: str,
    variants_txt_path: str,
    img_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    idx_to_label = build_idx_to_label_from_variants_txt(variants_txt_path)
    num_classes = len(idx_to_label)

    model = AircraftVariantCNN_v5(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")  # raw state_dict
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, idx_to_label, preprocess


def _ensure_variant_loaded(checkpoint_path: str, variants_txt_path: str, img_size: int = 224):
    global _VARIANT_MODEL, _VARIANT_IDX_TO_LABEL, _VARIANT_PREPROCESS
    if _VARIANT_MODEL is None:
        _VARIANT_MODEL, _VARIANT_IDX_TO_LABEL, _VARIANT_PREPROCESS = load_variant_model_state_dict(
            checkpoint_path=checkpoint_path,
            variants_txt_path=variants_txt_path,
            img_size=img_size
        )


def predict_aircraft_variant_top3(
    image_path: str,
    checkpoint_path: str = "./aircraft_varient.pt",
    variants_txt_path: str = "./variants.txt",
    img_size: int = 224,
):
    """
    Returns:
    {
      "pred_top1": str,
      "top3": [{"label": str, "confidence": float}, ...]
    }
    """
    _ensure_variant_loaded(checkpoint_path, variants_txt_path, img_size)

    model = _VARIANT_MODEL
    idx_to_label = _VARIANT_IDX_TO_LABEL
    preprocess = _VARIANT_PREPROCESS

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top3_prob, top3_idx = torch.topk(probs, k=3)

    top3_idx = top3_idx.cpu().tolist()
    top3_prob = top3_prob.cpu().tolist()

    top3 = [{"label": idx_to_label[int(i)], "confidence": float(p)} for i, p in zip(top3_idx, top3_prob)]
    return {"pred_top1": top3[0]["label"], "top3": top3}
