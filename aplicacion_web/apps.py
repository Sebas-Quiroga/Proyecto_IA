import logging
from pathlib import Path

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AplicacionWebConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "aplicacion_web"

    def ready(self):
        """Comprueba el modelo al iniciar sin descargarlo ni cargarlo."""
        from django.conf import settings

        model_path = Path(settings.MODEL_PATH)
        if model_path.exists():
            logger.info("Modelo encontrado: %s", model_path)
        else:
            logger.warning(
                "Modelo no encontrado en %s. Se descargará al primer análisis.",
                model_path,
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
