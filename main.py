from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from face_engine import ReconocedorFacial

app = FastAPI(title="Face Service - MediaPipe")

recognizer = ReconocedorFacial()

@app.post("/register-face")
async def register_face(file: UploadFile = File(...)):
    img_bytes = await file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return {"success": False, "message": "Invalid image"}

    ok = recognizer.registrar(frame)

    return {
        "success": ok,
        "message": "Face registered" if ok else "No face detected"
    }

@app.post("/verify-face")
async def verify_face(file: UploadFile = File(...)):
    img_bytes = await file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return {"verified": False}

    return recognizer.verificar(frame)
