from ultralytics import YOLO

# Load trained model
model = YOLO(
    "runs/yolov8_project-2/weights/best.pt"
)

# Predict
results = model.predict(
    source="dataset/valid/images/marine-debris-21-_JPG.rf.cc501a911315e9c9a43adfab28657e14.jpg",
          # replace with your image
    save=True,
    conf=0.25
)

# Print predictions
for result in results:
    boxes = result.boxes

    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        print(
            f"Class: {model.names[cls]}"
            f" | Confidence: {conf:.2f}"
        )