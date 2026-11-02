import cv2
from ultralytics import YOLO

yolo = YOLO("yolo26n.pt")
train_results = yolo.train(
    data="../datasets/GARBAGE CLASSIFICATION/data.yaml",
    epochs=1,
    imgsz=640,
    device="cpu"
)
metrics = yolo.val()

with open("metrics.txt", "w") as f:
    f.write(metrics)
    f.write(str(dir(metrics)))

with open("train_results.txt", "w") as f:
    f.write(train_results)
    f.write(str(dir(train_results)))
