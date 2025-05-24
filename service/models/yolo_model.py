from ultralytics import YOLO
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
model = YOLO("./yolo_weights/best.pt")
model.to(device)

part_names = list(model.names.values())

def predict_parts(img_bgr):
    results = model.predict(source=img_bgr, imgsz=640)
    masks = results[0].masks.data.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    part_names_for_masks = [part_names[i] for i in class_ids]
    return masks, part_names_for_masks