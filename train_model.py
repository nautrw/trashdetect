from datetime import timedelta
from timeit import default_timer as timer

from ultralytics import YOLO


def main():
    start_time = timer()

    yolo = YOLO("yolo26n.pt")
    yolo.train(
        data="./dataset.yaml",
        epochs=50,
        imgsz=640,
        # device="cpu",
        device=0,
        batch=16,
        workers=4,
    )

    path = yolo.export(format="onnx")

    end_time = timer()
    print(f"Model exported to {path}")
    print(f"Elapsed: {timedelta(seconds=end_time - start_time)}")


if __name__ == "__main__":
    main()
