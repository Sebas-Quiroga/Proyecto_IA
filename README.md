# Proyecto IA BERT — Clasificador de Currículums

Sistema web que **clasifica currículums (CVs) en 42 categorías profesionales** usando un modelo de red neuronal **BERT** fine-tuned (`bert-base-uncased`). El usuario sube su CV en PDF y la aplicación predice la categoría profesional con su nivel de confianza.

Pipeline completo: extracción de texto (PyMuPDF/OCR) → traducción → preprocesamiento NLP → clasificación BERT → resultado web.

## Características

- **Framework web:** Django 4.2 + plantillas con Chart.js
- **Modelo:** BERT (`bert-base-uncased`, ~110M parámetros) fine-tuned para 42 clases
- **OCR:** PyMuPDF (PDFs digitales) + Tesseract (PDFs escaneados)
- **NLP:** spaCy (NER, lemmatización), deep-translator + langdetect
- **Carga del modelo:** automática — descarga desde Google Drive si no existe; si falla, la app **no predice con pesos aleatorios**
- **Métricas reales:** `metrics.json` se genera al entrenar (accuracy, F1, matriz de confusión y series por época)
- **Despliegue:** Docker en Render (Gunicorn + WhiteNoise + Tesseract incluido) — ver [docs/07-despliegue-render.md](docs/07-despliegue-render.md)
- **Tests:** `pytest` para extracción de secciones, skills, limpieza y rutas

## Estructura del proyecto

```
Proyecto_IA_BERT/
├── manage.py                       # Punto de entrada Django
├── proyecto_cv/                    # Configuración del proyecto Django
│   ├── settings.py                 # Settings (media, static, WhiteNoise)
│   └── urls.py                     # URLs raíz
├── aplicacion_web/                 # Aplicación web
│   ├── views.py                    # Vistas: subida, análisis, info del modelo
│   ├── urls.py                     # Rutas: /, /upload/, /analyze/, /modelo/, /health/
│   └── templates/analyzer/         # Plantillas HTML + Chart.js
├── red_neuronal/                   # Núcleo de IA
│   ├── src/
│   │   ├── models/bert_classifier.py   # Clasificador BERT
│   │   ├── datasets/                   # Carga y preprocesamiento del dataset
│   │   ├── preprocessing/              # Limpieza NLP y extracción de skills
│   │   └── utils/config.py             # Configuración global
│   ├── train_bert_classifier.py    # Entrenamiento
│   ├── predict.py                  # Predicción por consola
│   └── models/                     # .pt + label_mapping.json + metrics.json
├── datos_entrenamiento/            # Dataset: training_data.csv (10,000 CVs)
├── pdfs_temporales/                # PDFs de usuarios (borrado automático)
├── tests/                          # Pruebas unitarias con pytest
├── Dockerfile · docker-entrypoint.sh · render.yaml
├── build.sh                        # Build alternativo sin Docker
├── .env.example                    # Variables de entorno documentadas
└── docs/                           # Documentación completa
```

## Requisitos

- Python 3.11 (el `Dockerfile` usa `python:3.11-slim`)
- Tesseract OCR (solo para PDFs escaneados)
- Dependencias: `requirements.txt` (raíz). `red_neuronal/requirements.txt` se mantiene por compatibilidad

## Instalación rápida

```bash
git clone https://github.com/Sebas-Quiroga/Proyecto_IA.git
cd Proyecto_IA

# Entorno virtual
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

# Dependencias
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Variables de entorno (opcional en local)
copy .env.example .env            # Windows
# cp .env.example .env            # Linux/Mac

# Base de datos y arranque
python manage.py migrate
python manage.py runserver
```

> En Windows instala Tesseract y define `TESSERACT_CMD` si no está en el PATH.

Guía detallada: [docs/01-instalacion.md](docs/01-instalacion.md)

## Ejecución

La aplicación queda disponible en `http://127.0.0.1:8000/`

| Ruta | Descripción |
|------|-------------|
| `/` | Página principal: subir CV en PDF |
| `/upload/` | Procesa el CV y redirige al inicio (patrón PRG) |
| `/analyze/` | Procesa el CV y muestra el resultado en la misma página |
| `/modelo/` | Información del modelo con gráficas interactivas |
| `/health/` | Health check (estado del modelo) para Render |

**Flujo:** subir PDF → extraer texto (PyMuPDF/OCR) → traducir al inglés → limpiar con NLP → BERT predice categoría → resultado con confianza % → PDF temporal eliminado.

## Modelo de Machine Learning

| Aspecto | Valor |
|---------|-------|
| Arquitectura | BERT-base-uncased (12 capas Transformer) |
| Parámetros | ~110 millones |
| Tarea | Clasificación de texto (multi-clase) |
| Categorías | 42 profesiones |
| Dataset | 10,000 currículums (`training_data.csv`) |
| Entrenamiento | 3 épocas · batch 16 · lr 2e-5 · split 80/20 · early stopping |
| Precisión | Según `metrics.json` (el valor histórico de referencia era ~89%) |

El modelo se entrena con transfer learning: se ajusta el BERT pre-entrenado con una capa clasificadora propia (768 → 42). El entrenamiento usa semillas fijas, `class_weight="balanced"`, early stopping y guarda el mejor checkpoint. Detalles completos en [docs/03-red-neuronal-bert.md](docs/03-red-neuronal-bert.md).

### Artefactos generados

| Archivo | Contenido |
|---------|-----------|
| `red_neuronal/models/bert_classifier_category.pt` | Pesos del modelo y metadatos |
| `red_neuronal/models/label_mapping.json` | Mapeo `categoría → índice` usado en entrenamiento |
| `red_neuronal/models/metrics.json` | Accuracy, F1, reporte, matriz de confusión y curvas por época |

El `label_mapping.json` debe viajar junto al `.pt`: reconstruir el mapeo desde el CSV puede cambiar el orden de las clases.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q aplicacion_web proyecto_cv red_neuronal tests
```

## Solución de problemas

| Error | Causa/Solución |
|-------|----------------|
| `ModuleNotFoundError: No module named 'src'` | Agregar `red_neuronal/` al PYTHONPATH (ya configurado en settings) |
| `TesseractNotFoundError` | Instalar Tesseract y definir `TESSERACT_CMD` (variable de entorno) |
| `No such file: training_data.csv` | Verificar dataset en `datos_entrenamiento/` |
| Modelo no descarga | Descarga manual desde Google Drive (ver docs/01) |
| «El modelo BERT no está disponible» | Revisar `MODEL_PATH`, `MODEL_MIN_SIZE` y la descarga; la app falla de forma segura en vez de predecir con pesos aleatorios |
| Predicciones sin sentido | Verificar que `label_mapping.json` corresponde al modelo entrenado |

Más detalles: [docs/01-instalacion.md §7](docs/01-instalacion.md)

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [01-instalacion.md](docs/01-instalacion.md) | Instalación, configuración y solución de problemas |
| [02-arquitectura-tecnologias.md](docs/02-arquitectura-tecnologias.md) | Pipeline, tecnologías y justificación |
| [03-red-neuronal-bert.md](docs/03-red-neuronal-bert.md) | Documentación técnica completa del modelo BERT |
| [04-aplicacion-web.md](docs/04-aplicacion-web.md) | Vistas, rutas y flujo de la aplicación web |
| [05-datos-entrenamiento.md](docs/05-datos-entrenamiento.md) | Dataset y distribución de categorías |
| [06-metricas-graficas.md](docs/06-metricas-graficas.md) | Métricas del modelo y gráficas |
| [07-despliegue-render.md](docs/07-despliegue-render.md) | Despliegue en producción (Render) |
| [documentacion-completa.html](docs/documentacion-completa.html) | Manual técnico completo (HTML) |
| [manual-tecnico.pdf](docs/manual-tecnico.pdf) | Manual técnico completo (PDF) |
