FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user \
 && mkdir -p /home/user/app \
 && chown user:user /home/user/app
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
USER user

# Vehicle detector (YOLOX-S, OpenCV model zoo); checksum-verified by the script.
RUN python scripts/fetch_model.py

# Demo limits: small uploads, capped processing, short retention.
ENV MT_DATA_DIR=/home/user/app/data \
    MT_MAX_UPLOAD_MB=50 \
    MT_MAX_DURATION_S=60 \
    MT_MAX_PROCESS_SIZE=960 \
    MT_MAX_FRAMES=600 \
    MT_RETENTION_HOURS=1 \
    MT_DEMO_NOTE="Demo limits: up to 50 MB and 60 s per video. Uploads and results are deleted after 1 hour."

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
