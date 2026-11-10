# Despliegue en Render

Guía de despliegue de la aplicación en [Render](https://render.com) usando el **runtime Docker**. La imagen incluye Tesseract, por lo que el OCR funciona igual que en local.

---

## Archivos de configuración

| Archivo | Propósito |
|---------|-----------|
| `render.yaml` | Blueprint del servicio web (runtime Docker, disco y variables) |
| `Dockerfile` | Imagen `python:3.11-slim` + Tesseract + dependencias |
| `docker-entrypoint.sh` | Migraciones, verificación del modelo y arranque de Gunicorn |
| `build.sh` | Alternativa de build sin Docker (útil para debugging) |

---

## Configuración del servicio (`render.yaml`)

```yaml
services:
  - type: web
    name: cv-bert-analyzer
    runtime: docker
    dockerfilePath: ./Dockerfile
    plan: starter
    healthCheckPath: /health/
    envVars:
      - key: DJANGO_DEBUG
        value: "False"
      - key: SECRET_KEY
        generateValue: true
      - key: MODEL_PATH
        value: /app/data/models/bert_classifier_category.pt
      - key: LABEL_MAPPING_PATH
        value: /app/data/models/label_mapping.json
      - key: METRICS_PATH
        value: /app/data/models/metrics.json
      - key: MODEL_GOOGLE_DRIVE_ID
        value: "1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um"
      - key: OCR_LANG
        value: eng+spa
    disk:
      name: cv-bert-data
      mountPath: /app/data
      sizeGB: 1
```

El disco persistente en `/app/data` evita descargar el modelo (~440 MB) en cada arranque o redeploy.

---

## Proceso de arranque

`docker-entrypoint.sh`:

1. Ejecuta `python manage.py migrate --noinput`.
2. Verifica/descarga el modelo BERT con `ensure_model_exists()`.
3. Arranca Gunicorn en `0.0.0.0:${PORT:-10000}` con workers/threads configurables.

El build de la imagen:

1. Instala Tesseract (`tesseract-ocr`, `eng` y `spa`).
2. Instala **torch CPU-only** desde el índice oficial.
3. Instala `requirements.txt` y `en_core_web_sm`.
4. Ejecuta `collectstatic` (WhiteNoise).

---

## Ajustes de Django para producción

`proyecto_cv/settings.py` se controla por variables de entorno:

```python
DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() == "true"
SECRET_KEY = os.environ["SECRET_KEY"]        # obligatoria si DEBUG=False
ALLOWED_HOSTS = os.environ["ALLOWED_HOSTS"].split(",")
CSRF_TRUSTED_ORIGINS = os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",")

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = True                   # con DEBUG=False
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

`RENDER_EXTERNAL_HOSTNAME` se añade automáticamente a `ALLOWED_HOSTS` y `CSRF_TRUSTED_ORIGINS`.

---

## Variables de entorno recomendadas

| Variable | Valor en Render |
|----------|-----------------|
| `DJANGO_DEBUG` | `False` |
| `SECRET_KEY` | Generada por Render (`generateValue: true`) |
| `ALLOWED_HOSTS` | Tu dominio (`*.onrender.com` o dominio propio) |
| `MODEL_PATH` | `/app/data/models/bert_classifier_category.pt` |
| `LABEL_MAPPING_PATH` | `/app/data/models/label_mapping.json` |
| `METRICS_PATH` | `/app/data/models/metrics.json` |
| `OCR_LANG` | `eng+spa` |
| `WEB_CONCURRENCY` | `1` (BERT en CPU consume ~1.5 GB RAM) |
| `WEB_THREADS` | `2` |
| `WEB_TIMEOUT` | `180` |

> No subas el `.pt` al repositorio: `MODEL_GOOGLE_DRIVE_ID` lo descarga y el disco lo conserva.

---

## Despliegue manual

1. Conectar el repositorio en el dashboard de Render.
2. Crear un **Web Service** con runtime **Docker**.
3. Render detecta `render.yaml` (Blueprint) o configura:
   - **Dockerfile Path:** `./Dockerfile`
   - **Health Check Path:** `/health/`
4. Añadir las variables de entorno de la tabla anterior.
5. Añadir un disco persistente montado en `/app/data`.

---

## Verificación

```bash
curl https://TU-SERVICIO.onrender.com/health/
# {"status": "ok", "model_exists": true, "model_loaded": false}
```

`model_loaded` pasa a `true` tras el primer análisis (carga lazy).

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| Build falla instalando torch | Se instala antes con el índice CPU-only; revisar logs |
| `TesseractNotFoundError` | La imagen ya lo incluye; verificar que se despliegue el Dockerfile correcto |
| Modelo no descarga | Revisar `MODEL_GOOGLE_DRIVE_ID`, cuota de Drive y logs |
| `502` al arrancar | Aumentar `WEB_TIMEOUT`; BERT tarda en cargar en CPU |
| Reinicio descarga el modelo | Confirmar que el disco está montado en `/app/data` |
| CSRF falla en producción | Añadir el dominio a `ALLOWED_HOSTS` y `CSRF_TRUSTED_ORIGINS` |
