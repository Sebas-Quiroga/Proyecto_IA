# Imagen de producción con Tesseract incluido.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=proyecto_cv.settings

WORKDIR /app

# Tesseract es necesario para PDFs escaneados.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --only-binary :all: \
        "tokenizers>=0.13,<1.0" "transformers>=4.30,<5.0" \
    && python -m pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

# Genera el manifiesto de WhiteNoise (DEBUG=False + SECRET_KEY temporal).
RUN DJANGO_DEBUG=False SECRET_KEY=build-only \
    python manage.py collectstatic --noinput

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 10000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
