#!/usr/bin/env bash

set -euo pipefail

echo "=== INICIANDO BUILD ==="

MODEL_PATH="${MODEL_PATH:-red_neuronal/models/bert_classifier_category.pt}"
MODEL_DIR="$(dirname "$MODEL_PATH")"

echo "1. Actualizando pip..."
python -m pip install --upgrade pip

echo "2. Instalando torch CPU-only..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "3. Instalando tokenizers/transformers con wheels binarios..."
python -m pip install --only-binary :all: "tokenizers>=0.13,<1.0" "transformers>=4.30,<5.0"

echo "4. Instalando dependencias..."
python -m pip install -r requirements.txt

echo "5. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm

echo "6. Preparando directorio del modelo..."
mkdir -p "$MODEL_DIR"

echo "7. Descargando modelo BERT (si no existe)..."
if [ ! -s "$MODEL_PATH" ]; then
  python -c "
import gdown
url = 'https://drive.google.com/uc?id=${MODEL_GOOGLE_DRIVE_ID:-1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um}'
gdown.download(url, '${MODEL_PATH}', quiet=False)
"
fi

if [ ! -s "$MODEL_PATH" ]; then
  echo "ERROR: no se pudo descargar el modelo BERT." >&2
  exit 1
fi

echo "8. Recolectando archivos estáticos..."
DJANGO_DEBUG=False SECRET_KEY=build-only python manage.py collectstatic --noinput

echo "9. Ejecutando migraciones..."
python manage.py migrate --noinput

echo "=== BUILD COMPLETADO ==="
