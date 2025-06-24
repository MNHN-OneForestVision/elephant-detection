FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y python3-tk ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "ui_yolo.py"]
