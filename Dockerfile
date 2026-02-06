FROM python:3.11-slim

WORKDIR /app

# Instalamos las librerías de sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Instalamos las dependencias (esto instalará la versión gráfica de OpenCV)
RUN pip install --no-cache-dir -r requirements.txt

# 2. EL TRUCO: Quitamos la versión gráfica y forzamos la versión de servidor (Headless)
# Esto soluciona el error de "no attribute solutions"
RUN pip uninstall -y opencv-contrib-python && \
    pip install opencv-python-headless

COPY . .

# Comando de inicio
CMD ["sh", "-c", "uvicorn main_websocket:app --host 0.0.0.0 --port ${PORT:-8080}"]