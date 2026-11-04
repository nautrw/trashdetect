import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(0)

yolo = YOLO("./runs/detect/train-2/weights/best.onnx")

while True:
    ret, frame = camera.read()
    results = yolo(frame, conf=0.5)

    annotated_frame = next(iter(results)).plot()

    for result in results:
        print(result)

    cv2.imshow('Camera', annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break
