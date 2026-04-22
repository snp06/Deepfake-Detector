import cv2, numpy as np
from tensorflow.keras.models import load_model

model = load_model("model_xception.h5", compile=False)
IMG = 299
MAX_FRAMES = 30

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x,y,w,h = faces[0]
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (IMG, IMG))
    return face / 255.0

def predict_video(path):
    cap = cv2.VideoCapture(path)
    preds = []
    step = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // MAX_FRAMES, 1)
    i = 0

    while cap.isOpened() and len(preds) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        if i % step == 0:
            f = preprocess_face(frame)
            if f is not None:
                p = model.predict(np.expand_dims(f, 0), verbose=0)[0][0]
                preds.append(p)
        i += 1
    cap.release()

    if not preds:
        return "Error", 0.0

    preds = np.array(preds)
    avg = preds.mean()
    if avg > 0.6:
        return "REAL", float(avg)
    elif avg < 0.4:
        return "FAKE", float(1-avg)
    else:
        return "UNCERTAIN", float(1 - preds.std())