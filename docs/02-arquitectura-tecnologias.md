# Arquitectura y Tecnologías

Descripción técnica de las tecnologías del proyecto, su rol en el pipeline y la justificación de cada elección.

---

## 1. Pipeline Completo del Sistema

```
Usuario sube PDF
    ↓
Django recibe petición (views.py)
    ↓
PyMuPDF (texto embebido) + fallback Tesseract (PDF escaneado)
    ↓
GoogleTranslator (traducción al inglés)
    ↓
spaCy (NER, lemmatización, extracción de secciones)
    ↓
TextCleaner (limpieza de texto)
    ↓
BERT tokenizer (convertir texto a números)
    ↓
Modelo BERT (predicción de categoría)
    ↓
Django renderiza resultado en HTML
    ↓
Usuario ve la predicción
```

---

## 2. Django (Framework Web)

Framework web full-stack de Python que maneja toda la **interfaz web**:

- Servidor web que recibe peticiones de los usuarios
- Rutas (URLs) hacia las vistas
- Plantillas HTML con los resultados
- Gestión de archivos subidos (PDFs)
- Base de datos (SQLite)

### Componentes usados

| Componente | Función en el proyecto |
|------------|------------------------|
| `django.shortcuts.render` | Renderiza plantillas HTML con datos |
| `django.core.files.storage` | Guarda archivos subidos (PDFs) |
| `django.conf.settings` | Configuración global |
| `django.urls.path` | Define rutas/URLs |
| `django.views` | Funciones que procesan peticiones |

### ¿Por qué Django y no Flask?

| Característica | Flask | Django |
|----------------|-------|--------|
| Tipo | Micro-framework | Full-stack |
| Base de datos | No incluida | ORM incluido |
| Autenticación | Manual | Incluida |
| Estructura | Flexible | Opinada (opinionated) |
| Tamaño de proyecto | Pequeño-mediano | Grande |

Para este proyecto Django aporta más estructura lista, mejor manejo de seguridad de archivos subidos y mejor integración con templates.

---

## 3. Dependencias de IA y Procesamiento

### Deep Learning y Modelos de Lenguaje

**torch (PyTorch)** — Framework de deep learning de Meta. Permite entrenar redes neuronales mediante diferenciación automática y retropropagación. Usado para definir, entrenar y evaluar el modelo BERT.

**transformers (Hugging Face)** — Librería con arquitecturas pre-entrenadas de última generación (BERT, RoBERTa, DistilBERT). Incluye tokenizadores, configuraciones y métodos de fine-tuning. Es el estándar de la industria para NLP.

### Ciencia de Datos

**pandas** — Manipulación y análisis de datos con DataFrames optimizados. Carga y preprocesa el dataset de currículums.

**numpy** — Computación numérica: arrays multidimensionales y funciones matemáticas de alto rendimiento.

**scikit-learn** — ML clásico: métricas (accuracy, F1-score), validación cruzada, split de datos y evaluación.

### NLP y Texto

**spaCy** — Framework industrial de NLP:
- NER: reconocimiento de entidades (personas, organizaciones, fechas)
- POS tagging: categorías gramaticales
- Lemmatización: reducción de palabras a su raíz
- Dependency parsing: análisis sintáctico

Modelo usado: `en_core_web_sm`.

**deep-translator** — Traducción automática del español al inglés antes de procesar (el modelo está entrenado en inglés).

**langdetect** — Detección automática de idioma (algoritmo n-gram); evita traducir textos ya en inglés.

### Extracción de Texto de PDFs

**pymupdf (fitz)** — Extrae texto de PDFs digitales (con texto embebido) y convierte páginas a imágenes para OCR.

**pytesseract** — Wrapper de Python para Tesseract OCR (motor de Google). Convierte imágenes a texto; requiere Tesseract instalado en el sistema.

**Pillow** — Procesamiento de imágenes: convierte páginas PDF a imágenes antes de pasarlas a Tesseract.

### Utilidades y despliegue

**gdown** — Descarga del modelo BERT desde Google Drive cuando no existe en disco.

**gunicorn** — Servidor WSGI de producción.

**whitenoise** — Sirve archivos estáticos directamente desde Django (sin nginx).

**Docker** — Imagen de producción con Tesseract incluido y disco persistente para el modelo.

> La configuración se lee de variables de entorno del sistema (`.env.example` documenta las disponibles). Django no carga `.env` automáticamente.

---

## 4. Resumen: Rol por Tecnología

| Tecnología | Propósito |
|------------|-----------|
| Django | Servidor web y estructura |
| PyMuPDF | Leer PDFs digitales |
| Tesseract + pytesseract | Leer PDFs escaneados (OCR) |
| Pillow | Manipular imágenes |
| GoogleTranslator | Traducción automática |
| langdetect | Detección de idioma |
| spaCy | Procesamiento de lenguaje natural |
| BERT + transformers | Modelo de machine learning |
| torch | Framework de deep learning |
| pandas / scikit-learn | Manejo y evaluación del dataset |
| gdown | Descarga del modelo entrenado |
| gunicorn / whitenoise | Servidor y estáticos en producción |
| Docker | Empaquetado y despliegue con Tesseract |

Cada tecnología cumple un rol específico en el pipeline de extremo a extremo: extraer → traducir → limpiar → analizar → clasificar.
