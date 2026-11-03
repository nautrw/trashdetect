from ultralytics import YOLO

def main():
    yolo = YOLO("yolo26n.pt")
    yolo.train(
        data="../datasets/GARBAGE CLASSIFICATION/data.yaml",
        epochs=1,
        imgsz=640,
        # device="cpu",
        device=0,
        batch=16,
        workers=4
    )

    path = yolo.export(format='onnx')
    print(f"Model exported to {path}")

if __name__ == "__main__":
    main()
