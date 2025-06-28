# download_models.py
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

print("Downloading YOLOv8...")
YOLO('yolov8n.pt')

print("Downloading CLIP model...")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Models downloaded successfully!")