#!/usr/bin/env sh

set -eu

echo "Ejecutando migraciones..."
python manage.py migrate --noinput

echo "Verificando modelo BERT..."
python -c "
import django
django.setup()
from aplicacion_web.views import ensure_model_exists
import sys
sys.exit(0 if ensure_model_exists() else 1)
"

echo "Iniciando Gunicorn..."
exec gunicorn proyecto_cv.wsgi:application \
    --bind "0.0.0.0:${PORT:-10000}" \
    --workers "${WEB_CONCURRENCY:-1}" \
    --threads "${WEB_THREADS:-2}" \
    --timeout "${WEB_TIMEOUT:-180}" \
    --access-logfile - \
    --error-logfile -
