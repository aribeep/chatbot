import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch, r=16)

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
            nn.MaxPool2d(2)  # 224 -> 56
        )
        self.stage1 = nn.Sequential(ResidualSEBlock(64, 64), ResidualSEBlock(64, 64))
        self.stage2 = nn.Sequential(ResidualSEBlock(64, 128, stride=2), ResidualSEBlock(128, 128))
        self.stage3 = nn.Sequential(ResidualSEBlock(128, 256, stride=2), ResidualSEBlock(256, 256))
        self.stage4 = nn.Sequential(ResidualSEBlock(256, 512, stride=2), ResidualSEBlock(512, 512))

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x)

def _ensure_loaded():
    global _MODEL, _IDX_TO_LABEL, _PREPROCESS
    if _MODEL is None:
        _MODEL, _IDX_TO_LABEL, _PREPROCESS = load_aircraft_model("./aircraft_family_best.pt")

model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()
def default_layer_picker(m):
    # Prefer stage3 (higher-res) if available; otherwise stage4; otherwise last conv-ish module.
    if hasattr(m, "stage3"):
        return m.stage3[-1]
    if hasattr(m, "stage4"):
        return m.stage4[-1]
    if hasattr(m, "features") and len(list(m.features)) > 0:
        return m.features[-1]
    # fallback: last module (not ideal, but avoids crashing)
    return list(m.modules())[-1]


target_layer = default_layer_picker(model)

    # -------------------------
    # Grad-CAM hooks storage
    # -------------------------
activations = {"value": None}
gradients   = {"value": None}

def fwd_hook(module, inp, out):
    activations["value"] = out

def bwd_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0]

    # register hooks
h1 = target_layer.register_forward_hook(fwd_hook)
h2 = target_layer.register_full_backward_hook(bwd_hook)

    # -------------------------
    # Load image + forward
    # -------------------------
img_pil = Image.open(image_path).convert("RGB")
x = preprocess(img_pil).unsqueeze(0).to(device)  # (1,3,H,W)

with torch.no_grad():
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).squeeze(0)
    top3_prob, top3_idx = torch.topk(probs, k=3)

    top3_idx  = top3_idx.detach().cpu().numpy().tolist()
    top3_prob = top3_prob.detach().cpu().numpy().tolist()

    def idx_to_name(i):
        if isinstance(idx_to_label, dict):
            return idx_to_label[int(i)]
            return str(i)
        top3 = [{"label": idx_to_name(i), "confidence": float(p)} for i, p in zip(top3_idx, top3_prob)]
        pred_top1 = top3[0]["label"]
        pred_idx  = int(top3_idx[0])

        model.zero_grad(set_to_none=True)
        logits2 = model(x)                       # forward again (tracked)
        score   = logits2[:, pred_idx].sum()
        score.backward(retain_graph=False)

        acts = activations["value"]              # (1,C,h,w)
        grads = gradients["value"]               # (1,C,h,w)

            # compute CAM
        weights = grads.mean(dim=(2,3), keepdim=True)        # (1,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)      # (1,1,h,w)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()        # (H,W) in [0,1]

        # -------------------------
        # Overlay CAM on original image
        # -------------------------
        cam_img = Image.fromarray(np.uint8(cam_np * 255)).resize(img_pil.size, resample=Image.BILINEAR)
        cam_np2 = np.array(cam_img).astype(np.float32) / 255.0

        heat = np.zeros((cam_np2.shape[0], cam_np2.shape[1], 3), dtype=np.float32)
        heat[..., 0] = cam_np2
        heat[..., 1] = cam_np2 * 0.2
        heat[..., 2] = (1.0 - cam_np2) * 0.2

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        overlay = (1 - alpha) * img_np + alpha * heat
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)

        # -------------------------
        # Save overlay
        # -------------------------
        os.makedirs(save_dir, exist_ok=True)
        out_name = f"gradcam_{uuid.uuid4().hex}.png"
        out_path = os.path.join(save_dir, out_name)
        overlay_pil.save(out_path)

        # cleanup hooks
        h1.remove()
        h2.remove()

        return {
            "pred_top1": pred_top1,
            "top3": top3,
            "gradcam_path": out_path
        }
# Example: your model class might ignore dropout_p; that's fine.
result = predict_aircraft_variant_with_cam(
    image_path="some_aircraft.jpg",
    model_builder=lambda num_classes, dropout_p: AircraftVariantCNN_v5(num_classes=num_classes)
)

print(result["pred_top1"])
print(result["top3"])
print(result["gradcam_path"])