# Guía de Instalación y Configuración

Pasos para instalar las dependencias e inicializar el sistema de clasificación de currículums.

---

## Requisitos Previos

| Requisito | Versión | Notas |
|-----------|---------|-------|
| Python | 3.11 (recomendado) | El `Dockerfile` usa `python:3.11-slim` |
| pip | Última | Gestor de paquetes de Python |
| Tesseract OCR | 5.x | Necesario solo si se procesan PDFs escaneados |
| Git | Opcional | Para clonar el repositorio |

---

## 1. Estructura del Proyecto

```
Proyecto_IA_BERT/
├── manage.py                  # Punto de entrada Django
├── proyecto_cv/               # Configuración del proyecto Django
├── aplicacion_web/            # Aplicación web (vistas, URLs, templates)
├── red_neuronal/              # Modelo BERT y NLP
│   ├── src/                   # Código fuente del modelo
│   ├── models/                # .pt + label_mapping.json + metrics.json
│   └── requirements.txt       # Compatibilidad (se usa requirements.txt raíz)
├── datos_entrenamiento/       # Dataset (training_data.csv)
├── pdfs_temporales/           # PDFs subidos por usuarios (se eliminan solos)
├── tests/                     # Pruebas con pytest
├── Dockerfile / docker-entrypoint.sh / render.yaml
├── build.sh                   # Build alternativo sin Docker
├── .env.example               # Variables de entorno documentadas
└── docs/                      # Documentación del proyecto
```

---

## 2. Instalar Dependencias

### 2.1 Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Dependencias del proyecto

```bash
# Dependencias de runtime (Django, torch, transformers, spaCy, OCR, etc.)
pip install -r requirements.txt

# Dependencias de desarrollo (pytest, ruff)
pip install -r requirements-dev.txt
```

Las dependencias principales son: `Django`, `torch`, `transformers`, `pandas`, `scikit-learn`, `spacy`, `pytesseract`, `deep-translator`, `langdetect`, `pymupdf`, `Pillow`, `gdown`, `gunicorn`, `whitenoise`. Ver [02-arquitectura-tecnologias.md](02-arquitectura-tecnologias.md) para el rol de cada una.

### 2.2.1 Variables de entorno

Copia `.env.example` a `.env` y ajusta los valores. Las más importantes:

| Variable | Uso |
|----------|-----|
| `DJANGO_DEBUG` | `True` en local, `False` en producción |
| `SECRET_KEY` | Obligatoria cuando `DJANGO_DEBUG=False` |
| `ALLOWED_HOSTS` | Hosts permitidos, separados por coma |
| `MODEL_PATH` | Ruta del `.pt` |
| `LABEL_MAPPING_PATH` | Ruta de `label_mapping.json` |
| `METRICS_PATH` | Ruta de `metrics.json` |
| `MAX_PDF_SIZE` / `MAX_PDF_PAGES` | Límites de subida y OCR |
| `OCR_LANG` | Idiomas de Tesseract (ej. `eng+spa`) |
| `TESSERACT_CMD` | Ruta al ejecutable de Tesseract |
| `MODEL_GOOGLE_DRIVE_ID` | ID de descarga del modelo |
| `MODEL_SHA256` | Checksum opcional del modelo |

Django **no** lee el `.env` automáticamente: expórtalo en la shell o configúralo en el panel de Render.

### 2.3 Descargar modelo de spaCy

```bash
python -m spacy download en_core_web_sm
```

### 2.4 Instalar Tesseract OCR (solo PDFs escaneados)

**Windows:**
1. Descargar desde: <https://github.com/UB-Mannheim/tesseract/wiki>
2. Instalar y agregar al PATH del sistema
3. Opcional: instalar datos de idioma español (`tesseract-ocr-spa`)

**Linux (Ubuntu):**
```bash
sudo apt-get update && sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

**Mac (Homebrew):**
```bash
brew install tesseract
```

---

## 3. Configurar el Proyecto Django

El proyecto ya viene configurado; estos pasos son de verificación:

### 3.1 settings.py (`proyecto_cv/settings.py`)

```python
INSTALLED_APPS = [..., 'aplicacion_web']

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'pdfs_temporales'

# Rutas del modelo configurables por entorno
MODEL_PATH = BASE_DIR / 'red_neuronal' / 'models' / 'bert_classifier_category.pt'
LABEL_MAPPING_PATH = MODEL_PATH.parent / 'label_mapping.json'
METRICS_PATH = MODEL_PATH.parent / 'metrics.json'

# Ruta del núcleo de IA al PYTHONPATH
sys.path.insert(0, str(BASE_DIR / 'red_neuronal'))
```

### 3.2 Crear carpetas generadas en runtime

```bash
mkdir pdfs_temporales          # PDFs temporales de usuarios
mkdir red_neuronal/models      # Modelo y artefactos
```

---

## 4. Ejecutar el Proyecto

```bash
# Migraciones
python manage.py makemigrations
python manage.py migrate

# Servidor de desarrollo
python manage.py runserver
```

La aplicación queda disponible en: <http://127.0.0.1:8000/>

> **Nota PowerShell:** si Django no encuentra el módulo de settings, ejecutar antes:
> `$env:DJANGO_SETTINGS_MODULE="proyecto_cv.settings"`

---

## 5. Modelo BERT

Al iniciar, la aplicación verifica si existe `red_neuronal/models/bert_classifier_category.pt`:

1. **Si existe y es válido** → se carga directamente.
2. **Si NO existe** → se descarga automáticamente desde Google Drive usando `gdown` (implementado en `aplicacion_web/views.py`).
3. **Si la descarga falla** → la vista devuelve un error controlado. La app **no** predice con pesos aleatorios.

La validación comprueba tamaño mínimo (`MODEL_MIN_SIZE`) y, si está definido, `MODEL_SHA256`. La descarga usa un archivo `.part` y reemplazo atómico.

### Descarga manual (alternativa)

**Enlace:** <https://drive.google.com/file/d/1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um/view?usp=sharing>

Guardar como: `red_neuronal/models/bert_classifier_category.pt`

> El modelo debe ir acompañado de `label_mapping.json`. Sin él, la app intenta reconstruir el mapeo desde el CSV y registra una advertencia; el orden de clases podría no coincidir con el entrenamiento.

### Entrenamiento local (opcional)

```bash
cd red_neuronal
python train_bert_classifier.py
```

Genera tres artefactos:

- `models/bert_classifier_category.pt` — pesos y metadatos
- `models/label_mapping.json` — mapeo `categoría → índice`
- `models/metrics.json` — accuracy, F1, reporte, matriz de confusión y curvas por época

Requiere el dataset en `datos_entrenamiento/1_resume_classification/training_data.csv`. Ver [05-datos-entrenamiento.md](05-datos-entrenamiento.md).

---

## 6. Verificar la Instalación

1. Abrir <http://127.0.0.1:8000/>
2. Subir un PDF de currículum
3. El sistema debe mostrar la categoría profesional predicha y su confianza

---

## 7. Solución de Problemas

### `ModuleNotFoundError: No module named 'src'`

Agregar la ruta del núcleo de IA al `PYTHONPATH` (ya incluida en `settings.py`):

```python
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'red_neuronal'))
```

### `TesseractNotFoundError`

Configurar la ruta con la variable de entorno `TESSERACT_CMD` (no editar `views.py`):

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

En Linux/Docker el valor por defecto es `/usr/bin/tesseract`.

### `No such file or directory: training_data.csv`

Verificar que el dataset exista en `datos_entrenamiento/1_resume_classification/training_data.csv`.

### «El modelo BERT no está disponible»

Revisar `MODEL_PATH`, `MODEL_MIN_SIZE` y la descarga desde Google Drive. Este error es intencional: evita devolver predicciones aleatorias cuando el modelo no cargó.

### Predicciones incoherentes

Verificar que `label_mapping.json` sea el que se generó junto con el `.pt`. Si se reconstruye desde el CSV, el orden de las clases puede cambiar.

---

## 8. Resumen de Comandos

```bash
venv\Scripts\activate                 # Activar entorno (Windows)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python manage.py migrate
python manage.py runserver

# Tests
pip install -r requirements-dev.txt
python -m pytest -q
```
