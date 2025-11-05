import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


MODEL_PATH = "cnn_model_fer.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (48, 48)

EMOTION_MAP = {
    0: "surprise",
    1: "fear",
    2: "disgust",
    3: "happiness",
    4: "sadness",
    5: "anger",
    6: "neutral"
}

def predict_emotion(face_img):
    face_resized = cv2.resize(face_img, IMG_SIZE)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_norm = face_rgb.astype("float32") / 255.0
    x = np.expand_dims(face_norm, axis=0)
    preds = model.predict(x, verbose=0)[0]
    idx = np.argmax(preds)
    emotion = EMOTION_MAP.get(idx, "unknown")
    conf = preds[idx]
    return emotion, conf


st.title("😊 Facial Emotion Recognition App")
st.write("Upload a face image to detect emotion using your trained CNN model (RAF dataset).")

uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Please upload a clearer image.")
    else:
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            emotion, conf = predict_emotion(face)

            # Vẽ khung + label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion} ({conf*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Emotion", use_column_width=True)
