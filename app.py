import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO(
    "runs/yolov8_project-2/weights/best.pt"
)

st.title("Marine Debris Detection using YOLOv8")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image"
    )

    results = model.predict(
        image,
        conf=0.25
    )

    result_image = results[0].plot()

    st.image(
        result_image,
        caption="Detected Marine Debris"
    )

    st.subheader("Detections")

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        st.write(
            f"{model.names[cls]} : {conf:.2f}"
        )