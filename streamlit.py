import streamlit as st
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import torch

# ---------------- SETTINGS ----------------
MODEL_PATHS = ["best.pt"]  # You can add multiple models if needed
CONF_THRESHOLD = 0.10      # Lower confidence for minor fracture sensitivity
IMG_SIZE = 1024            # Higher resolution improves small object detection
# -------------------------------------------

st.set_page_config(page_title="Hand Fracture Detection", layout="centered")
st.title("Hand Fracture Detection App")
st.write("Upload a hand X-ray image — this AI model will detect even small or hairline fractures.")

@st.cache_resource
def load_models(paths):
    """Load one or more YOLO models"""
    models = []
    for p in paths:
        try:
            models.append(YOLO(p))
        except Exception as e:
            st.warning(f"⚠️ Could not load model {p}: {e}")
    return models

models = load_models(MODEL_PATHS)

uploaded_file = st.file_uploader("📤 Upload Hand X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Optional enhancement (improves minor fracture visibility)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)

    st.image(image, caption="Enhanced X-ray Image", use_column_width=True)

    if st.button("Detect Fractures"):
        with st.spinner("Detecting fractures... Please wait."):
            for idx, model in enumerate(models):
                # st.subheader(f"🧠 Model {idx+1} Prediction ({MODEL_PATHS[idx]})")

                # Run inference (no saving to file)
                results = model.predict(
                    source=image,
                    conf=CONF_THRESHOLD,
                    imgsz=IMG_SIZE,
                    iou=0.3,   # lower IoU to allow overlapping detections (minor ones)
                    save=False,
                    show=False,
                    verbose=False
                )

                # Plot annotated results
                annotated_img = results[0].plot()  # numpy array with bounding boxes
                st.image(annotated_img, caption=f"Model {idx+1} Detection Result", use_column_width=True)

                # Detection info
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    st.success(f"✅ {len(boxes)} fracture(s) detected.")
                    for i, box in enumerate(boxes):
                        cls = int(box.cls)
                        conf = float(box.conf)
                        xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
                        st.write(f"🩹 Fracture #{i+1}: Confidence {conf*100:.2f}% at {xyxy}")
                else:
                    st.warning("⚠️ No visible fracture detected. Try adjusting brightness or uploading a clearer X-ray.")

else:
    st.info("📸 Please upload a hand X-ray image to start detection.")

# st.markdown("---")
# st.caption("Developed by Tayyaba — AI-powered Hand Fracture Detection (Minor & Major) using YOLOv8 + Streamlit 🧠")
