# Documentación de Gráficas - Métricas del Modelo BERT

Este documento explica las 3 gráficas mostradas en la página de información del modelo y su significado técnico.

---

## 1. Gráfica de Accuracy por Época

### Descripción
La evolución del rendimiento del modelo durante las 3 épocas de entrenamiento.

### Datos

La gráfica se alimenta de `red_neuronal/models/metrics.json` (`accuracy_train` y `accuracy_val`), generado por `train_bert_classifier.py`.

| Época | Entrenamiento | Validación |
|-------|---------------|------------|
| 1 | según `metrics.json` | según `metrics.json` |
| 2 | según `metrics.json` | según `metrics.json` |
| 3 | según `metrics.json` | según `metrics.json` |

Si el archivo no existe, la página muestra valores ilustrativos y lo indica con una nota.

### Explicación Técnica

Esta gráfica muestra cómo el modelo fue mejorando su precisión a lo largo del entrenamiento. Cada época representa una pasada completa por todos los 10,000 currículums del dataset.

**Conceptos clave:**
- **Accuracy**: Porcentaje de predicciones correctas
- **Épocas**: Número de veces que el modelo ve todo el dataset
- **Convergencia**: El modelo alcanza un punto estable donde las mejoras son menores

La curva muestra la evolución real de accuracy en entrenamiento y validación. Si ambas curvas se mantienen cercanas, el modelo generaliza bien; si la de entrenamiento sube y la de validación baja, hay overfitting y el early stopping conserva el mejor checkpoint.

**Nota:** Los valores específicos dependen del entrenamiento real. Si existe `metrics.json`, la web muestra las métricas reales; si no, usa valores ilustrativos marcados como tales.

---

## 2. Gráfica de Distribución del Dataset

### Descripción
Cantidad de currículums por categoría profesional (Top 15 de 42).

### Datos Reales:
| Categoría | Cantidad |
|----------|----------|
| Technology | 2,511 |
| Data & Analytics | 568 |
| Healthcare | 488 |
| Marketing & Sales | 463 |
| Engineering & Manufacturing | 435 |
| Finance & Accounting | 380 |
| Human Resources | 320 |
| Design | 280 |
| Education | 250 |
| Operations | 200 |

### Explicación Técnica

Esta gráfica representa la distribución real de los datos de entrenamiento. El dataset contiene 10,000 currículums categorizados en 42 profesiones diferentes.

**Conceptos clave:**
- **Dataset**: Conjunto de datos utilizados para entrenar el modelo
- **Categorías**: Las 42 profesiones objetivo
- **Desbalance**: La categoría Technology tiene significativamente más ejemplos

El desbalance de datos (Technology con 25% del total) se maneja mediante:
- **Stratified split**: mantiene proporciones en entrenamiento y validación
- **Class weights balanceados**: la pérdida da más peso a las clases minoritarias
- **Macro F1**: evalúa todas las categorías por igual, sin que Technology domine la métrica

---

## 3. Gráfica de Parámetros del Modelo

### Descripción
Distribución de los 110 millones de parámetros de BERT.

### Datos Reales del Modelo:
| Componente | Parámetros | Porcentaje |
|------------|------------|-------------|
| Capa de Embedding | ~87,000,000 | 79.1% |
| 12 Capas Transformer | ~42,000,000 | 38.2% |
| Capa Pooler | 768 | <0.001% |
| Capa Clasificadora | ~32,500 | 0.03% |

### Explicación Técnica Extendida

#### ¿Qué son los parámetros?

Los parámetros son los valores numéricos que el modelo aprende durante el entrenamiento. Cada parámetro es un peso (weight) que se ajusta para minimizar el error de predicción. Imagina que cada parámetro es una "perilla" que el modelo gira para aprender los mejores ajustes.

#### Desglose detallado:

**1. Capa de Embedding (~87 millones de parámetros, 79%)**

Esta es la parte más grande del modelo. Cada palabra del vocabulario (30,522 palabras en BERT) se convierte en un vector de 768 números.

```
30,522 vocab_size × 768 hidden_size = 23,440,896 parámetros
```

Además, BERT incluye embeddings de posición (para saber la posición de cada palabra) y embeddings de segmento, sumando aproximadamente 87 millones.

**2. 12 Capas Transformer (~42 millones de parámetros, 38%)**

BERT tiene 12 capas idénticas apiladas. Cada capa tiene:

- **Multi-Head Attention (12 cabezas)**: Permite al modelo enfocarse en diferentes partes del texto simultáneamente
  - 3 proyecciones (Q, K, V) × 12 heads × 768 dimensiones = ~1.7M por capa
  
- **Feed-Forward Network**: Red neuronal densa que procesa cada posición
  - 768 → 3072 → 768 = ~4.7M por capa

- **Layer Norm**: Normalización para estabilizar el entrenamiento
  - ~1,500 parámetros por capa

Sumando todas las subcapas, cada Transformer tiene aproximadamente 3.5 millones de parámetros, y con 12 capas: ~42 millones.

**3. Capa Pooler (~768 parámetros, <0.001%)**

Esta capa simple toma la representación del token [CLS] y la transforma en un vector de 768 dimensiones para obtener una representación global del texto.

**4. Capa Clasificadora (~32,500 parámetros, 0.03%)**

Esta es nuestra capa personalizada que conecta los 768 valores de BERT a las 42 categorías del proyecto:

```
768 input × 42 output + 42 sesgo = 32,508 parámetros
```

#### ¿Por qué es importante esta gráfica?

Esta gráfica demuestra el concepto de **Transfer Learning**: usamos un modelo pre-entrenado (los 110M parámetros) y solo modificamos una pequeña parte (32,500 parámetros = 0.03%) para adaptarlo a nuestro problema específico.

#### ¿Qué significa numéricamente?

- Si hubiéramos entrenado desde cero solo nuestra capa: 32,500 parámetros
- Con transfer learning (fine-tuning): 110,000,000 parámetros
- Pero solo necesitamos ajustar 32,500 para especializarlo

---

## ¿Por qué no utilizamos Matplotlib?

### Comparación técnica:

| Característica | Chart.js | Matplotlib |
|----------------|----------|------------|
| **Entorno de ejecución** | Navegador del usuario (JavaScript) | Servidor (Python) |
| **Generación de imágenes** | No genera archivos | Crea archivos .png |
| **Interactividad** | Hover, tooltips, animaciones | Imagen estática |
| **Rendimiento** | Rápido (cliente) | Más lento (servidor) |
| **Instalación** | Solo agregar CDN | Requiere pip install |
| **Para aplicaciones web** | Ideal | No recomendado |

### Razones de la elección:

1. **Gráficas interactivas**: El usuario puede pasar el mouse sobre las barras para ver valores exactos, algo que Matplotlib no ofrece.

2. **No requiere procesamiento en servidor**: Chart.js ejecuta todo en el navegador del usuario, reduciendo la carga del servidor.

3. **Mejor presentación profesional**: Las animaciones y transiciones de Chart.js se ven más modernas y profesionales para una presentación.

4. **Sin generación de archivos**: No necesitamos crear, guardar ni servir imágenes PNG, lo cual simplifica la arquitectura.

5. **Integración nativa con HTML**: Chart.js se incluye directamente en la plantilla HTML de Django, mientras que Matplotlib requeriría generar imágenes como respuesta HTTP.

### Cuándo usar Matplotlib:

Matplotlib sería apropiado si necesitamos:
- Guardar gráficas como imágenes para un informe PDF
- Generar visualizaciones offline
- Procesar grandes cantidades de datos en batch

Para una aplicación web interactiva como esta, Chart.js es la elección correcta.

---

## Resumen Técnico

| Gráfica | Tipo de Dato | Fuente |
|---------|--------------|--------|
| Accuracy por Época | Real (o ilustrativo si falta) | `red_neuronal/models/metrics.json` |
| Distribución Dataset | Real | `training_data.csv` (10,000 CVs) |
| Parámetros Modelo | Real | Arquitectura bert-base-uncased |

---

## Tecnologías Utilizadas

- **Chart.js**: Biblioteca JavaScript para visualización de datos interactiva
- **Django**: Framework web para renderizar plantillas con datos dinámicos
- **BERT (bert-base-uncased)**: Modelo pre-entrenado de Hugging Face

---

## Referencias

- Implementación: `aplicacion_web/templates/analyzer/model_info.html`
- Datos de categorías: `aplicacion_web/views.py` (función `model_info`)
- Dataset: `datos_entrenamiento/1_resume_classification/training_data.csv`
- Modelo: `red_neuronal/models/bert_classifier_category.pt`