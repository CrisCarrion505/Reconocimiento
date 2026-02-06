FROM python:3.11-slim

WORKDIR /app

# Actualizamos paquetes e instalamos libgl1 en lugar de libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando de inicio
CMD ["sh", "-c", "uvicorn main_websocket:app --host 0.0.0.0 --port ${PORT:-8080}"]