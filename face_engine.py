import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

class ReconocedorFacial:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.embedding_registrado = None

    def extraer_embedding(self, imagen):
        imagen = cv2.resize(imagen, (640, 480))
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado = self.face_mesh.process(rgb)

        if not resultado.multi_face_landmarks:
            return None

        puntos = resultado.multi_face_landmarks[0]
        coords = []
        for lm in puntos.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        return np.array(coords)

    def registrar(self, frame):
        embedding = self.extraer_embedding(frame)
        if embedding is None:
            return False
        self.embedding_registrado = embedding
        return True

    def verificar(self, frame):
        if self.embedding_registrado is None:
            return {
                "verified": False,
                "reason": "No face registered"
            }

        embedding_actual = self.extraer_embedding(frame)
        if embedding_actual is None:
            return {
                "verified": False,
                "reason": "No face detected"
            }

        distancia = np.linalg.norm(self.embedding_registrado - embedding_actual)

        return {
            "verified": bool(distancia < 2.0),
            "distance": float(distancia)
        }

    