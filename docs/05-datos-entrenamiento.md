# Datos de Entrenamiento

Dataset utilizado para entrenar el modelo BERT de clasificación de currículums.

---

## Ubicación

```
datos_entrenamiento/
└── 1_resume_classification/
    └── training_data.csv    # ÚNICO dataset usado
```

## Características del dataset

- **10,000 currículums** categorizados
- **42 categorías** profesionales
- **Split:** 80% entrenamiento / 20% validación (estratificado)

### Columnas principales

| Columna | Contenido |
|---------|-----------|
| `Resume Text` | Texto del currículum (educación, experiencia, skills) |
| `Category` | Categoría profesional objetivo |
| `Job Role` | Rol específico (referencia, no usado en este modelo) |

---

## Distribución de categorías (Top 10)

```
Technology                      2511  (25.1%)
Data & Analytics                 568   (5.7%)
Healthcare                       488   (4.9%)
Marketing & Sales                463   (4.6%)
Engineering & Manufacturing      435   (4.4%)
Finance & Accounting             380   (3.8%)
Human Resources                  320   (3.2%)
... (35 categorías más)
```

El desbalance (Technology concentra el 25%) se maneja mediante:
- **Stratified split**: mantiene proporciones en entrenamiento y validación
- **Class weights balanceados**: `compute_class_weight("balanced")` en la pérdida
- **Macro F1 y weighted F1**: métricas complementarias a accuracy

---

## Hiperparámetros de entrenamiento usados con este dataset

| Parámetro | Valor |
|-----------|-------|
| Épocas | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Split | 80/20 estratificado |
| Warmup | 10% de los pasos |
| Early stopping | paciencia 2 |
| Semilla | 42 |
| Class weights | `balanced` |

Ver [03-red-neuronal-bert.md](03-red-neuronal-bert.md) para el proceso completo.

---

## Uso en el código

```python
from src.datasets.data_loader import DatasetLoader

loader = DatasetLoader()
df = loader.load_training_data()

texts  = df['Resume Text'].tolist()
labels = df['Category'].tolist()
```

## Notas

- **Solo este dataset** es necesario para entrenar el modelo BERT; otros CSV fueron eliminados por no usarse.
