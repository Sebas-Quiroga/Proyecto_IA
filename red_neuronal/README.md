# Sistema de Clasificación de Currículums con BERT

Proyecto de IA para clasificar currículums en categorías profesionales usando BERT.

---

## Estructura de Carpetas

```
red_neuronal/
├── main.py                      # Punto de entrada
├── predict.py                   # Predicción con modelo entrenado
├── train_bert_classifier.py     # Entrenamiento del modelo
├── requirements.txt             # Dependencias
│
├── src/
│   ├── preprocessing/           # Limpieza de texto
│   │   ├── text_cleaner.py     # Limpiador de texto NLP
│   │   ├── skill_extractor.py # Detector de habilidades
│   │   └── nlp_processor.py   # Pipeline NLP completo
│   │
│   ├── datasets/               # Carga y preprocesamiento
│   │   ├── data_loader.py      # Carga training_data.csv
│   │   └── preprocessor.py     # Prepara datos para BERT
│   │
│   ├── models/                 # Modelos de ML
│   │   └── bert_classifier.py # Clasificador BERT
│   │
│   └── utils/                  # Configuración
│       └── config.py           # Rutas y parámetros
│
├── models/                     # .pt + label_mapping.json + metrics.json
└── datos_entrenamiento/        # Dataset (training_data.csv)
```

---

## Cómo Ejecutar

### 1. Instalar Dependencias
```bash
cd red_neuronal
pip install -r requirements.txt
```

### 2. Entrenar el Modelo
```bash
python train_bert_classifier.py
```

El entrenamiento genera:
- `models/bert_classifier_category.pt` — pesos y metadatos
- `models/label_mapping.json` — mapeo `categoría → índice`
- `models/metrics.json` — accuracy, F1, reporte y curvas por época

### 3. Hacer Predicciones
```bash
python predict.py
```

### 4. Prueba rápida del pipeline (opcional)
```bash
python scripts/smoke_train.py
```

---

## Módulos del Proyecto

| Módulo | Función |
|--------|---------|
| `data_loader.py` | Carga training_data.csv (10,000 CVs) |
| `preprocessor.py` | Limpia datos y divide train/val |
| `bert_classifier.py` | Modelo BERT para clasificación |
| `text_cleaner.py` | Limpia texto del CV |
| `skill_extractor.py` | Extrae habilidades del CV |
| `nlp_processor.py` | Pipeline NLP completo |

---

## Modelo Entrenado

El modelo BERT entrenado se encuentra disponible en Google Drive:

**Enlace:** https://drive.google.com/file/d/1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um/view?usp=sharing

### Descarga del modelo:
1. Acceder al enlace de Google Drive
2. Descargar el archivo `bert_classifier_category.pt`
3. Colocar el archivo en la carpeta `red_neuronal/models/`
4. Colocar `label_mapping.json` en la misma carpeta (se genera al entrenar)

**Modelo entrenado:**
- Archivo: `bert_classifier_category.pt`
- Precisión: ~89% en validación
- Categorías: 42 clases profesionales

---

## Dataset

- **training_data.csv**: 10,000 currículums con 42 categorías
- Ubicación: `datos_entrenamiento/1_resume_classification/training_data.csv`
- Uso: 80% entrenamiento, 20% validación

---

## Notas

- El modelo fue entrenado con 3 épocas
- Solo se usa 1 dataset (training_data.csv)
- El modelo entrenado debe descargarse desde Google Drive