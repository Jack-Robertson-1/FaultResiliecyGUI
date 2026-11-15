import torch
from PIL import Image
import numpy as np
from ultralytics.data.augment import LetterBox
from torchvision.ops import batched_nms
from models import device

_letterbox = LetterBox((640, 640), auto=False, stride=32)

def load_image_tensor(path):
    """Load an image and preprocess it into a model-ready tensor."""
    img = np.array(Image.open(path).convert("RGB"))
    img = _letterbox(image=img)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

def decode_output(raw_output, model):
    """Turn raw model output into a list of (class, confidence) detections."""
    detect_layer = model.model[-1]
    preds = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    preds = preds.transpose(1, 2)
    out = detect_layer.postprocess(preds)[0]
    mask = out[:, 4] > 0.25
    out = out[mask]
    if out.shape[0] == 0:
        return []

    # xywh -> xyxy, nms() needs corner coords not center/width/height
    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    scores = out[:, 4]
    classes = out[:, 5]
    # per-class NMS - boxes of different classes shouldn't suppress each other,
    # only overlapping boxes of the SAME class count as duplicates. Using plain
    # nms() here was wrongly dropping real detections whenever a different-class
    # box happened to overlap a higher-confidence one.
    keep = batched_nms(boxes_xyxy, scores, classes, iou_threshold=0.45)
    out = out[keep].detach().cpu().numpy()
    return [(int(row[5]), float(row[4])) for row in out]