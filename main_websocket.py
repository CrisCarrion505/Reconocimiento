from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime
import base64

# BORRA TODO y pega ESTO (código migrado que te di):
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime

app = FastAPI(title="Monitor Examen WebSocket")

# ✅ IMPORTS NUEVOS (MediaPipe Tasks) - REEMPLAZA línea 11
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image

face_landmarker = None

def init_mediapipe():
    global face_landmarker
    if face_landmarker is None:
        model_path = "/app/models/face_landmarker.task"
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )
        face_landmarker = FaceLandmarker.create_from_options(options)



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
        init_mediapipe()  # Inicializa antes del loop

        try:
            while True:
                data = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                resultado = self.analizar_frame(frame)
                await websocket.send_json(resultado)

        except WebSocketDisconnect:
            self.metrics["tiempo_total"] = (datetime.now() - self.inicio_examen).total_seconds()
            print("Examen terminado:", self.metrics)

    def analizar_frame(self, frame):
        self.metrics["frames_procesados"] += 1
        
        # Convertir a RGB y MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Procesar con FaceLandmarker
        results = face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            self.metrics["rostros_perdidos"] += 1
            return {"status": "rostro_perdido", **self.metrics}

        self.metrics["rostros_detectados"] += 1

        # Extraer landmarks (mismo índice que FaceMesh)
        landmarks = results.face_landmarks[0]  # Lista de NormalizedLandmark
        
        yaw = self.calcular_yaw(landmarks)
        self.yaw_history.append(yaw)

        desviacion = 0
        if len(self.yaw_history) > 1:
            avg_yaw = sum(self.yaw_history) / len(self.yaw_history)
            desviacion = abs(yaw - avg_yaw)
            if desviacion > 0.08:
                self.metrics["desvios_mirada"] += 1

        return {
            "status": "ok",
            "yaw": yaw,
            "desvacion": desviacion,
            **self.metrics
        }

    def calcular_yaw(self, landmarks):
        # Mismos índices que FaceMesh: nose_tip=1, left_eye=33, right_eye=362
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[362]

        eye_vector_x = right_eye.x - left_eye.x
        nose_vector_x = nose_tip.x - (left_eye.x + right_eye.x) / 2

        yaw = nose_vector_x / eye_vector_x if eye_vector_x != 0 else 0
        return yaw

@app.websocket("/ws/examen/{examen_id}")
async def websocket_examen(websocket: WebSocket, examen_id: str):
    await websocket.accept()
    print(f"Examen {examen_id} iniciado")

    monitor = MonitorExamen()
    await monitor.procesar_stream(websocket)

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Monitor Examen - Webcam Test</title>
        <style>
            video, canvas { border: 2px solid #333; margin: 10px; }
            #status { font-weight: bold; padding: 10px; margin: 10px; border-radius: 5px; }
            .ok { background: #d4edda; color: #155724; }
            .warning { background: #fff3cd; color: #856404; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>🧑‍💻 Monitor Examen - Webcam + WebSocket</h1>
        
        <div>
            <video id="webcam" width="640" height="480" autoplay muted></video>
            <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        </div>
        
        <div id="status">🔴 Desconectado - Click "INICIAR EXAMEN"</div>
        
        <div>
            <button onclick="iniciarExamen()">🚀 INICIAR EXAMEN</button>
            <button onclick="detenerExamen()">⏹️ DETENER EXAMEN</button>
        </div>
        
        <div id="metrics"></div>
        
        <script>
            let ws = null;
            let stream = null;
            let examenId = 'test-123';
            const statusEl = document.getElementById('status');
            const metricsEl = document.getElementById('metrics');
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            async function iniciarExamen() {
                try {
                    // 1. Pedir permisos webcam
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480 }
                    });
                    video.srcObject = stream;

                    // 2. Conectar WebSocket
                    const url = `wss://reconocimiento-1.onrender.com/ws/examen/${examenId}`;
                    ws = new WebSocket(url);
                    
                    ws.onopen = () => {
                        statusEl.textContent = '🟢 CONECTADO - Enviando frames...';
                        statusEl.className = 'ok';
                        enviarFrames();
                    };

                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        mostrarMetrics(data);
                    };

                    ws.onerror = () => {
                        statusEl.textContent = '🔴 ERROR WebSocket';
                        statusEl.className = 'error';
                    };

                } catch(err) {
                    statusEl.textContent = '❌ Error: ' + err.message;
                    statusEl.className = 'error';
                }
            }

            function enviarFrames() {
                if (!ws || ws.readyState !== WebSocket.OPEN || !stream) return;

                ctx.drawImage(video, 0, 0, 640, 480);
                canvas.toBlob((blob) => {
                    blob.arrayBuffer().then(buffer => {
                        ws.send(buffer);
                    });
                }, 'image/jpeg', 0.8);

                // 10 FPS
                setTimeout(enviarFrames, 100);
            }

            function mostrarMetrics(data) {
                const yaw = data.yaw?.toFixed(3) || 'N/A';
                const status = data.status || 'unknown';
                
                let color = 'ok';
                let emoji = '🟢';
                if (status === 'rostro_perdido') {
                    color = 'warning';
                    emoji = '🟡';
                }
                
                metricsEl.innerHTML = `
                    <div style="margin:10px;">
                        <strong>${emoji} Status:</strong> ${status}<br>
                        <strong>Yaw:</strong> ${yaw}<br>
                        <strong>Frames:</strong> ${data.frames_procesados || 0}<br>
                        <strong>Rostros:</strong> ${data.rostros_detectados || 0} / ${data.rostros_perdidos || 0}<br>
                        <strong>Desvíos:</strong> ${data.desvios_mirada || 0}
                    </div>
                `;
            }

            function detenerExamen() {
                if (ws) ws.close();
                if (stream) stream.getTracks().forEach(track => track.stop());
                statusEl.textContent = '🔴 Desconectado';
                statusEl.className = '';
                metricsEl.innerHTML = '';
            }
        </script>
    </body>
    </html>
    """)

