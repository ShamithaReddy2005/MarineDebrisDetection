from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="marine_debris",
        name="yolov8_project"
    )

if __name__ == "__main__":
    main()