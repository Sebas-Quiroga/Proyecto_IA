"""
Rutas de la aplicación web.

- /            Inicio: subir CV y ver el resultado.
- /upload/     Procesa el CV y redirige al inicio (PRG).
- /analyze/    Variante que renderiza el resultado en la misma página.
- /modelo/     Información del modelo y métricas.
- /health/     Health check para Render.
"""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_cv, name="upload_cv"),
    path("analyze/", views.analyze_pdf, name="analyze_pdf"),
    path("modelo/", views.model_info, name="model_info"),
    path("health/", views.health, name="health"),
]
