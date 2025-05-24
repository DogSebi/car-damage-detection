import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import cv2

device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
processor = SegformerImageProcessor.from_pretrained("./segformer_weights")
model = SegformerForSemanticSegmentation.from_pretrained("./segformer_weights").to(device)
model.eval()

def predict_damage(img_bgr):
    img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_mask = logits.argmax(dim=1)[0].cpu()
    resized_mask = torch.nn.functional.interpolate(
        pred_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(640, 640),
        mode="nearest"
    ).squeeze().long().numpy()
    return resized_mask
