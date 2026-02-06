FROM python:3.11-slim

WORKDIR /app

# 1. Instalamos librerías del sistema (necesarias para visión artificial)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 2. INSTALACIÓN CRÍTICA (Bloque único para evitar caché)
# - Instalamos las dependencias normales (que traerán la versión mala de OpenCV)
# - INMEDIATAMENTE borramos la versión mala (opencv-contrib)
# - Instalamos la versión buena (headless) para servidores
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-contrib-python opencv-python && \
    pip install opencv-python-headless

COPY . .

# 3. Comprobación de seguridad (Esto fallará el deploy si OpenCV está roto, avisándonos antes)
RUN python -c "import cv2; print('OpenCV cargado correctamente:', cv2.__version__)"

# 4. Comando de inicio
CMD ["sh", "-c", "uvicorn main_websocket:app --host 0.0.0.0 --port ${PORT:-8080}"]