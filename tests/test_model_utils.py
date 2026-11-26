import numpy as np
from PIL import Image
from model_utils import _build, predict

CFG = {"architecture": "densenet121", "image_size": 224,
       "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

def test_predict_valid_probabilities():
    model = _build("densenet121").eval()
    img = Image.fromarray((np.random.rand(300, 300) * 255).astype("uint8"))
    p = predict([img, img], model, CFG)
    assert p.shape == (2, 2)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)