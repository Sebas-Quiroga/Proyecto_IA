"""
Django settings for proyecto_cv.

La configuración de producción se controla por variables de entorno. En local,
si no se define ``DJANGO_DEBUG``, se asume modo desarrollo.
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


BASE_DIR = Path(__file__).resolve().parent.parent

# ====================================================================
# SEGURIDAD BÁSICA
# ====================================================================

DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() == "true"

SECRET_KEY = os.environ.get("SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured(
            "SECRET_KEY no está definida. Configúrala como variable de entorno."
        )
    SECRET_KEY = "dev-only-insecure-key-change-me"

RENDER_EXTERNAL_HOSTNAME = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if host.strip()
]
if DEBUG and not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["*"]
elif not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
if RENDER_EXTERNAL_HOSTNAME and RENDER_EXTERNAL_HOSTNAME not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]
if RENDER_EXTERNAL_HOSTNAME:
    CSRF_TRUSTED_ORIGINS.append(f"https://{RENDER_EXTERNAL_HOSTNAME}")

# Render termina TLS en su proxy.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

if not DEBUG:
    # Permite desactivarlo si el proxy no envía X-Forwarded-Proto.
    SECURE_SSL_REDIRECT = (
        os.environ.get("SECURE_SSL_REDIRECT", "True").lower() == "true"
    )
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 31_536_000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"


# ====================================================================
# APLICACIONES
# ====================================================================

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "aplicacion_web",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "proyecto_cv.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "proyecto_cv.wsgi.application"


# ====================================================================
# BASE DE DATOS
# ====================================================================

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("DATABASE_PATH", BASE_DIR / "db.sqlite3"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# ====================================================================
# INTERNACIONALIZACIÓN
# ====================================================================

LANGUAGE_CODE = "es"
TIME_ZONE = os.environ.get("TIME_ZONE", "America/Bogota")
USE_I18N = True
USE_TZ = True


# ====================================================================
# ARCHIVOS ESTÁTICOS Y MEDIA
# ====================================================================

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    # En desarrollo no se exige el manifiesto de collectstatic.
    "staticfiles": {
        "BACKEND": (
            "django.contrib.staticfiles.storage.StaticFilesStorage"
            if DEBUG
            else "whitenoise.storage.CompressedManifestStaticFilesStorage"
        )
    },
}

MEDIA_URL = "/media/"
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", BASE_DIR / "pdfs_temporales"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ====================================================================
# RUTAS DEL MODELO Y LÍMITES DE LA APLICACIÓN
# ====================================================================

MODEL_DIR = Path(
    os.environ.get("MODEL_DIR", BASE_DIR / "red_neuronal" / "models")
)
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH", MODEL_DIR / "bert_classifier_category.pt"
    )
)
LABEL_MAPPING_PATH = Path(
    os.environ.get("LABEL_MAPPING_PATH", MODEL_DIR / "label_mapping.json")
)
METRICS_PATH = Path(os.environ.get("METRICS_PATH", MODEL_DIR / "metrics.json"))
DATASET_PATH = Path(
    os.environ.get(
        "DATASET_PATH",
        BASE_DIR
        / "datos_entrenamiento"
        / "1_resume_classification"
        / "training_data.csv",
    )
)

# Descarga opcional del modelo.
MODEL_GOOGLE_DRIVE_ID = os.environ.get(
    "MODEL_GOOGLE_DRIVE_ID", "1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um"
)
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "")
MODEL_MIN_SIZE = int(os.environ.get("MODEL_MIN_SIZE", 10 * 1024 * 1024))

# Límites de subida y OCR.
MAX_PDF_SIZE = int(os.environ.get("MAX_PDF_SIZE", 5 * 1024 * 1024))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", 10))
OCR_LANG = os.environ.get("OCR_LANG", "eng")
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")

# Sesiones: los CVs son datos personales; no mantenerlos indefinidamente.
SESSION_COOKIE_AGE = int(os.environ.get("SESSION_COOKIE_AGE", 60 * 30))
SESSION_EXPIRE_AT_BROWSER_CLOSE = True


# ====================================================================
# LOGGING
# ====================================================================

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        }
    },
    "root": {"handlers": ["console"], "level": os.environ.get("LOG_LEVEL", "INFO")},
}


# ====================================================================
# PYTHONPATH PARA LOS MÓDULOS DE red_neuronal
# ====================================================================

import sys

sys.path.insert(0, str(BASE_DIR / "red_neuronal"))
