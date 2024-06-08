from ultralytics import YOLO
from icecream import ic
import cv2

model = YOLO("yolov8x-cls.pt")
model = YOLO("yolov8n-cls.pt")
# model = YOLO("yolov8m-cls.pt")
retrain = True

if retrain:
    results = model.train(data='./yolodata/', epochs = 200, imgsz = 90)
    model.export(format='onnx')
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    ic(metrics.top1)   # top1 accuracy
    ic(metrics.top5)   # top5 accuracy
    ic(metrics)

# HOW TO LOAD
# model = YOLO('./yolo/runs/classify/train6/weights/best.pt')
