import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(0)

yolo = YOLO("yolo26n.pt")
train_results = yolo.train(
    data="realwaste.yaml",
    epochs=100,
    imgsz=640,
    device="cpu"
)
metrics = yolo.val()

while True:
    ret, frame = camera.read()
    results = yolo(frame, conf=0.75)

    annotated_frame = next(iter(results)).plot()

    for result in results:
        print(result)

    cv2.imshow('Camera', annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break