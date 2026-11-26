import json
import numpy as np
import torch, torch.nn as nn
from torchvision import models, transforms

def _build(arch):
    if arch == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "densenet121":
        m = models.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, 2)
    else:
        raise ValueError(arch)
    return m

def load_model(weights_path="best_model.pt", config_path="config.json", device="cpu"):
    cfg = json.load(open(config_path))
    model = _build(cfg["architecture"])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval(), cfg

def get_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

def get_target_layer(model, cfg):
    return model.layer4[-1] if cfg["architecture"] == "resnet50" else model.features[-1]

@torch.no_grad()
def predict(images, model, cfg, device="cpu"):
    """images : liste d'images PIL. Retourne (N, 2) : [P(NORMAL), P(PNEUMONIA)]."""
    tf = get_transform(cfg)
    x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device)
    return torch.softmax(model(x), dim=1).cpu().numpy()