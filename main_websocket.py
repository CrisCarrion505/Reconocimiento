from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import mediapipe as mp
import json
import asyncio
from collections import deque
from datetime import datetime

app = FastAPI(title="Monitor Examen WebSocket")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

class MonitorExamen:
    def __init__(self):
        self.metrics = {
            "tiempo_total": 0,
            "frames_procesados": 0,
            "rostros_detectados": 0,
            "rostros_perdidos": 0,
            "desvios_mirada": 0
        }
        self.yaw_history = deque(maxlen=20)
        self.inicio_examen = None

    async def procesar_stream(self, websocket: WebSocket):
        self.inicio_examen = datetime.now()
        contador = 0
        
        try:
            while True:
                # Recibir frame como bytes
                data = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Procesar frame
                resultado = self.analizar_frame(frame)
                contador += 1
                
                # Enviar métricas cada frame
                await websocket.send_json(resultado)
                
        except WebSocketDisconnect:
            self.metrics["tiempo_total"] = (datetime.now() - self.inicio_examen).total_seconds()
            print("Examen terminado:", self.metrics)

    def analizar_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        self.metrics["frames_procesados"] += 1
        
        if not results.multi_face_landmarks:
            self.metrics["rostros_perdidos"] += 1
            return {"status": "rostro_perdido", **self.metrics}
        
        self.metrics["rostros_detectados"] += 1
        
        # Calcular yaw simple (ángulo cabeza)
        landmarks = results.multi_face_landmarks[0]
        yaw = self.calcular_yaw(landmarks)
        self.yaw_history.append(yaw)
        
        # Detectar desviación
        if len(self.yaw_history) > 1:
            avg_yaw = sum(self.yaw_history) / len(self.yaw_history)
            desviacion = abs(yaw - avg_yaw)
            if desviacion > 0.08:
                self.metrics["desvios_mirada"] += 1
        
        return {
            "status": "ok",
            "yaw": yaw,
            "desviacion": desviacion if 'desviacion' in locals() else 0,
            **self.metrics
        }
    
    def calcular_yaw(self, landmarks):
        # Landmarks clave: nariz y orejas
        nose_tip = landmarks.landmark[1]  # Punta nariz
        left_eye = landmarks.landmark[33]  # Ojo izquierdo
        right_eye = landmarks.landmark[362]  # Ojo derecho
        
        # Vector ojos
        eye_vector_x = right_eye.x - left_eye.x
        # Vector nariz centro
        nose_vector_x = nose_tip.x - (left_eye.x + right_eye.x) / 2
        
        yaw = nose_vector_x / eye_vector_x if eye_vector_x != 0 else 0
        return yaw

monitor = MonitorExamen()

@app.websocket("/ws/examen/{examen_id}")
async def websocket_examen(websocket: WebSocket, examen_id: str):
    await websocket.accept()
    print(f"Examen {examen_id} iniciado")
    await monitor.procesar_stream(websocket)

@app.get("/")
async def get():
    return HTMLResponse("""
    <h1>WebSocket Test - Monitor Examen</h1>
    <button onclick="testWebSocket()">Probar WebSocket</button>
    <script>
        async function testWebSocket() {
            const ws = new WebSocket('ws://localhost:8000/ws/examen/123');
            ws.onmessage = (event) => console.log(JSON.parse(event.data));
        }
    </script>
    """)
