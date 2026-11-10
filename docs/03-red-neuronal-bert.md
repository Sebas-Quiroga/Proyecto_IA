# DOCUMENTACIÓN TÉCNICA DE LA RED NEURONAL BERT
## Sistema de Clasificación de Currículums

> **Nota:** las referencias a números de línea son orientativas. El código fuente en `red_neuronal/` es la fuente de verdad.

---

# 1. INTRODUCCIÓN Y CONTEXTO

## 1.1 ¿Qué es una Red Neuronal?

Una **red neuronal artificial** es un modelo computacional inspirado en cómo funcionan las neuronas del cerebro biológico. Está compuesta por:

- **Neuronas** (nodos): unidades que procesan información
- **Capas**: grupos de neuronas que operan en paralelo
  - Capa de entrada: recibe los datos
  - Capas ocultas: procesan la información
  - Capa de salida: produce el resultado
- **Pesos y sesgos**: parámetros que se ajustan durante el entrenamiento

```
Entrada → [Capa 1] → [Capa 2] → ... → [Capa Salida] → Predicción
```

## 1.2 ¿Por qué usamos una red neuronal para este proyecto?

**Problema a resolver**: Clasificar automáticamente currículums en 42 categorías profesionales basándose en su contenido.

**Por qué no usar reglas simples?**
- Los currículums varían enormemente en formato y contenido
- Las categorías no son obvias (un "Python developer" podría estar en "Technology" o "Software")
- El lenguaje natural es complejo y ambiguo

**Por qué una red neuronal?**
- Aprende patrones complejos directamente de los datos
- Generaliza a ejemplos no vistos
- Puede capturar relaciones semánticas sutiles
- El texto de un CV tiene muchas características implícitas

---

# 2. ELECCIÓN DE LA ARQUITECTURA: BERT

## 2.1 ¿Qué es BERT?

**BERT** (Bidirectional Encoder Representations from Transformers) es una arquitectura de red neuronal desarrollada por Google en 2018. Es un **Transformer** - una arquitectura que revolucionó el procesamiento del lenguaje natural.

### ¿Por qué BERT y no otra arquitectura?

| Arquitectura | Ventajas | Desventajas | ¿Por qué no? |
|--------------|----------|--------------|---------------|
| **BERT** | Pre-entrenado, bidirectional,state-of-the-art | Más pesado | ✅ Elegido |
| LSTM/RNN | Secuencial, bueno para secuencias | Lento, no captura contexto largo | No escala bien |
| CNN (texto) | Rápido | No captura relaciones lejanas | No óptimo para NLP |
| Red simple | Rápido | Poco poder expresivo | No suficiente |

**Razones específicas de la elección:**

1. **Pre-entrenamiento**: BERT ya fue entrenado con grandes cantidades de texto en inglés, lo que significa que ya "entiende" el lenguaje. Solo necesitamos ajustarlo (fine-tuning) para nuestro problema específico.

2. **Bidireccional**: BERT considera el contexto tanto a izquierda como derecha de cada palabra, a diferencia de modelos unidireccionales. Esto es crucial para entender currículums donde el contexto importa.

3. **State-of-the-art**: En 2018 y hasta ahora, BERT y sus variantes dominan las tareas de clasificación de texto. Es el estándar de la industria.

4. **Transfer Learning**: Podemos tomar el conocimiento general de BERT y adaptarlo a nuestro problema específico con pocos datos de entrenamiento.

## 2.2 Arquitectura interna de BERT

```
BERT-base:
├── Vocabulario: 30,522 palabras
├── Embedding: 768 dimensiones
├── Capa de embedding → 768
├── 12 capas Transformer
│   ├── Multi-head attention (12 heads)
│   ├── Feed-forward (3072 dimensiones)
│   └── Layer normalization
├── Capa de pooling → 768
└── Capa de clasificación → 42 (nuestras categorías)
```

**Parámetros totales: ~110 millones**

### Detalle de las capas:

1. **Capa de Embedding**:
   - Convierte cada palabra en un vector de 768 números
   - Usa "Token Embeddings", "Segment Embeddings" y "Position Embeddings"
   
2. **12 Capas Transformer** (ocultas):
   - Cada capa tiene:
     - **Multi-head Attention**: 12 heads que atienden a diferentes partes del texto
     - **Feed Forward Network**: red de 3072 → 768 dimensiones
     - **Layer Normalization**: estabiliza el entrenamiento
     - **Residual Connections**: conecta capas directamente
   
3. **Capa de Pooling**:
   - Toma el output del último token [CLS]
   - Lo pasa por una capa densa y tangente hiperbólica

4. **Capa de Clasificación**:
   - Linear: 768 → 42 (nuestras categorías)
   - + Softmax para obtener probabilidades

---

# 3. ELECCIÓN DE LIBRERÍAS Y DEPENDENCIAS

## 3.1 PyTorch (torch)

### ¿Qué es?
**PyTorch** es el framework de deep learning de Meta (Facebook). Es una biblioteca que permite definir y entrenar redes neuronales de manera eficiente.

### ¿Por qué se usó?
- **Flexibilidad**: Define grafos computacionales dinámicos (mejor para investigación y desarrollo)
- **Debugging**: Puedes usar herramientas estándar de Python (pdb, print)
- **Comunidad**: Gran ecosistema y documentación
- **Integración con transformers**: La librería de HuggingFace está built-in para PyTorch

### Configuración en el proyecto:
```python
import torch
import torch.nn as nn

# Dispositivo: CPU o GPU (cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo se mueve al dispositivo
model = model.to(device)
```

> **Código fuente:** `red_neuronal/train_bert_classifier.py:9` y `red_neuronal/src/models/bert_classifier.py:196`

## 3.2 Transformers (Hugging Face)

### ¿Qué es?
**Transformers** es la librería de Hugging Face que proporciona implementaciones de modelos pre-entrenados (BERT, RoBERTa, GPT, etc.) y herramientas para NLP.

### ¿Por qué se usó?
- **Modelos pre-entrenados**: Acceso directo a `bert-base-uncased`
- **Tokenizers**: Herramienta optimizada para tokenizar texto
- **Fine-tuning**: Funciones para ajustar modelos a nuevas tareas

### Configuración:
```python
from transformers import BertTokenizer, BertModel

# Tokenizer para convertir texto a tokens
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Modelo pre-entrenado
bert_model = BertModel.from_pretrained('bert-base-uncased')
```

> **Código fuente:** `red_neuronal/train_bert_classifier.py:114-115` y `red_neuronal/src/models/bert_classifier.py:134,202`

### ¿Por qué bert-base-uncased?
- **base**: Tamaño medio (12 capas, 768 hidden)
- **uncased**: No diferencia mayúsculas/minúsculas (suficiente para nuestro caso)
- Alternativas: bert-large (más preciso pero más lento), distilbert (más rápido pero menos preciso)

## 3.3 spaCy

### ¿Qué es?
**spaCy** es una librería industrial de NLP que proporciona herramientas rápidas para procesamiento de texto.

### ¿Por qué se usó?
- **NER (Named Entity Recognition)**: Detectar personas, organizaciones, fechas
- **Lemmatización**: Reducir palabras a su forma raíz (running → run)
- **POS Tagging**: Etiquetar categorías gramaticales
- **Detección de secciones**: Extraer experience, education, skills del CV

### Alternativas consideradas:
| Librería | Pros | Contras | ¿Por qué no? |
|----------|------|---------|---------------|
| **spaCy** | Rápida, industrial, buen NER | Modelo pequeño por defecto | ✅ Elegida |
| NLTK | Clásica, educativa | Lenta, código antiguo | Fuera de producción |
| Stanford NLP | Precisa | Java-based, compleja | Overkill |

## 3.4 Tesseract (OCR)

### ¿Qué es?
**Tesseract** es un motor de OCR (Optical Character Recognition) desarrollado por Google. Lee texto de imágenes.

### ¿Por qué se usó?
El proyecto debe procesar **dos tipos de PDFs**:

1. **PDFs digitales** (con texto embebido):
   - PyMuPDF (fitz) extrae el texto directamente
   - No necesita OCR

2. **PDFs escaneados** (imágenes):
   - No tienen texto, son solo píxeles
   - Necesitamos OCR para "leer" la imagen
   - Tesseract convierte imagen → texto

### Pipeline implementado:
```
Página del PDF
    ↓
Convertir a imagen (2x escala)
    ↓
Tesseract OCR (eng + spa)
    ↓
Texto extraído
```

### ¿Por qué Tesseract y no otras opciones?

| OCR | Tipo | Costo | Precisión | ¿Por qué no? |
|-----|------|-------|-----------|---------------|
| **Tesseract** | Open source | Gratis | Buena | ✅ Elegido |
| Amazon Textract | Cloud | De pago | Excelente | Costo |
| Google Vision | Cloud | De pago | Excelente | Costo |
| EasyOCR | Deep learning | Gratis | Mejor que Tesseract | Más lento |

## 3.5 Deep Translator

### ¿Qué es?
Biblioteca de Python que proporciona una interfaz unificada para múltiples APIs de traducción.

### ¿Por qué se usó?
- Los currículums pueden estar en español
- El modelo BERT fue entrenado en inglés
- Necesitamos traducir antes de procesar

```python
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='es', target='en')
text_en = translator.translate(text)
```

> **Código fuente:** `red_neuronal/src/preprocessing/nlp_processor.py` (GoogleTranslator)

---

# 4. CONFIGURACIÓN DE LA RED NEURONAL

## 4.1 Arquitectura del Modelo

### Capa de Entrada
```python
# Tokenización
encoding = tokenizer(
    text,
    add_special_tokens=True,    # [CLS] y [SEP]
    max_length=512,               # Máx 512 tokens (límite de BERT)
    padding='max_length',         # Rellenar a 512
    truncation=True,              # Truncar si es más largo
    return_attention_mask=True,   # Máscara de atención
    return_tensors='pt'           # Tensores PyTorch
)
```

> **Código fuente:** `red_neuronal/train_bert_classifier.py:114-120` y `red_neuronal/src/models/bert_classifier.py:220-235`

**¿Por qué 512 tokens?**
- BERT tiene un límite máximo de 512 tokens por diseño
- 512 tokens ≈ ~2000-3000 palabras (suficiente para un CV)
- No podemos procesar más sin arquitecturas especiales (Longformer, etc.)

### Capas Ocultas
```python
class BertClassifier(nn.Module):
    def __init__(self, num_classes=42):
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)  # 30% dropout
        self.classifier = nn.Linear(768, num_classes)  # 768 → 42
```

> **Código fuente:** `red_neuronal/src/models/bert_classifier.py:125-175`

**¿Por qué 768 dimensiones?**
- Es el tamaño estándar de BERT-base
- Compromiso entre poder expresivo y velocidad

**¿Por qué 30% de dropout?**
- Evita overfitting (memorizar datos de entrenamiento)
- 0.3 es un valor estándar para fine-tuning de BERT

### Capa de Salida
```python
# 42 categorías profesionales
# Softmax convierte logits a probabilidades
probs = torch.softmax(outputs.logits, dim=1)
# Ejemplo: [0.85, 0.05, 0.03, ...] para 42 clases
```

> **Código fuente:** `red_neuronal/src/models/bert_classifier.py:185-195`

## 4.2 Parámetros de Entrenamiento

```python
# Configuración de entrenamiento (red_neuronal/src/utils/config.py)
EPOCHS = 3                     # Número de épocas
BATCH_SIZE = 16                # Muestras por batch
LEARNING_RATE = 2e-5           # 0.00002 (típico para BERT)
MAX_LEN = 512                  # Longitud máxima de tokens
NUM_WORKERS = 0                # Seguro en Windows y contenedores
RANDOM_SEED = 42               # Reproducibilidad
EARLY_STOPPING_PATIENCE = 2    # Épocas sin mejora antes de parar

# Optimizador
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Scheduler (learning rate con warmup del 10%)
total_steps = max(1, len(train_loader) * EPOCHS)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)
```

> **Código fuente:** `red_neuronal/train_bert_classifier.py:129-133` y `red_neuronal/src/models/bert_classifier.py:253`

### Explicación técnica del Optimizador y Scheduler

#### Optimizador: AdamW

```python
optimizer = AdamW(model.parameters(), lr=2e-5)
```

**AdamW (Adam with Weight Decay)** es el optimizador recomendado en el paper original de BERT. Combina:

1. **Adam (Adaptive Moment Estimation)**:
   - Mantiene dos promedios móviles: momentum (gradientes) y RMSProp (gradientes al cuadrado)
   - **Momentum**: ayuda a continuar en la dirección correcta de aprendizaje
   - **RMSProp**: adapta el learning rate por parámetro automáticamente
     - Parámetros con gradientes grandes → learning rate pequeño
     - Parámetros con gradientes pequeños → learning rate grande
   - Imagina una bola que rodar por una superficie irregular: el momentum la mantiene en movimiento y RMSProp ajusta la velocidad según la inclinación

2. **Weight Decay (Regularización L2)**:
   - Multiplica cada peso por un factor menor a 1 (ej: 0.99) en cada actualización
   - Penaliza pesos grandes → evita overfitting
   - Se aplica directamente a los pesos, no a los gradientes (por eso es "W" y no Adam clásico)

**¿Por qué lr=2e-5 (0.00002)?**
- Valor del paper original de BERT tras múltiples experimentos
- Es un learning rate muy bajo porque ya estamos usando un modelo pre-entrenado
- No queremos cambiar drásticamente los pesos que BERT ya aprendió en su entrenamiento masivo
- LR muy alto → el modelo diverge (explota gradientes)
- LR muy bajo → converge lentamente o no aprende

#### Scheduler: Learning Rate Lineal con Warmup

```python
total_steps = max(1, len(train_loader) * EPOCHS)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)
```

**¿Qué es un Scheduler?**
Es un programador que ajusta el learning rate durante el entrenamiento. En lugar de mantenerlo fijo, lo cambia según un cronograma.

**Warmup (precalentamiento)**:
- Comienza con un LR muy bajo (casi 0) al inicio
- Aumenta gradualmente hasta alcanzar el LR objetivo (2e-5)
- ¿Por qué? Al principio los pesos están randomizados, un LR alto causaría actualizaciones grandes e inestables
- Da tiempo al modelo de "estabilizarse" antes de aprender

**num_warmup_steps = 10% de los pasos**:
- El LR sube gradualmente durante el primer 10% del entrenamiento
- Después decae linealmente hasta el final
- Evita actualizaciones inestables al inicio del fine-tuning

**num_training_steps**:
- Total de pasos de entrenamiento = batches por época × épocas
- Con 10,000 muestras y batch_size=16: ~625 batches por época
- × 3 épocas = ~1,875 pasos totales

**¿Por qué reducir el LR al final?**
- Al acercarse al óptimo, quieres movimientos más precisos
- LR alto te hace "saltar" sobre el mínimo
- LR bajo permite converger suavemente

```
Con scheduler (linear decay):
LR: ╲___________________________
    ↑                          ↑
   inicio                    final
```

**Resumen del flujo en entrenamiento:**
```python
for step in range(total_steps):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Clip gradientes (evitar explosión numérica)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Actualizar pesos (AdamW ajusta LR por parámetro)
    optimizer.step()
    
    # Actualizar LR según schedule
    scheduler.step()
    
    # Limpiar gradientes para siguiente batch
    optimizer.zero_grad()
```

### ¿Por qué estos valores específicos?

**EPOCHS = 3**
- Con 10,000 muestras y 42 clases, 3 épocas son suficientes
- Más épocas → overfitting
- Menos épocas → underfitting
- Es un balance entre tiempo de entrenamiento y precisión

**BATCH_SIZE = 16**
- estándar para BERT en GPU con 8GB de VRAM
- En CPU, 16 es aceptable
- Batch más grande → más estable pero más memoria

**LEARNING_RATE = 2e-5**
- Es el valor recomendado en el paper original de BERT
- Learning rate muy alto → el modelo diverge
- Learning rate muy bajo → converge muy lentamente

**WEIGHT_DECAY = 0.01**
- Regularización para evitar overfitting
- AdamW incluye weight decay por defecto

---

# 5. PROCESO DE ENTRENAMIENTO

## 5.1 Pipeline de Entrenamiento

```
Dataset (10,000 CVs)
    ↓
Preprocesamiento
├── Limpieza de texto
├── Tokenización
├── Codificación de etiquetas (42 categorías)
└── Train/Validation split (80/20)
    ↓
Entrenamiento (hasta 3 épocas)
├── Pesos por clase (balanced)
├── Por cada batch:
│   ├── Forward pass: texto → BERT → logits
│   ├── Calcular pérdida (CrossEntropy ponderada)
│   ├── Backward pass (gradientes)
│   ├── Clip gradientes (max_norm=1)
│   ├── Actualizar pesos
│   └── Actualizar learning rate
├── Evaluar train y validación después de cada época
├── Guardar el mejor estado (mayor accuracy de validación)
└── Early stopping si no mejora en 2 épocas
    ↓
Guardar modelo (.pt) + label_mapping.json + metrics.json
```

## 5.2 Función de Pérdida

```python
# CrossEntropyLoss multiclase con pesos por clase
criterion = nn.CrossEntropyLoss(weight=class_weights)
loss = criterion(outputs, labels)
```

Los `class_weights` se calculan con `compute_class_weight("balanced")` para que las clases minoritarias pesen más y el modelo no ignore categorías poco frecuentes.

> **Código fuente:** `red_neuronal/train_bert_classifier.py:170` y `red_neuronal/src/models/bert_classifier.py:178`

**¿Por qué CrossEntropyLoss?**
- Estándar para clasificación multiclase
- Combina softmax + pérdida de entropía cruzada
- Penaliza fuertemente predicciones incorrectas

## 5.3 Optimizador (AdamW)

```python
# AdamW: Adam con Weight Decay
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,           # Learning rate bajo para BERT
    weight_decay=0.01   # Regularización L2
)
```

> **Código fuente:** `red_neuronal/train_bert_classifier.py:127-131` y `red_neuronal/src/models/bert_classifier.py:248-252`

### ¿Qué es AdamW?

**AdamW** es una versión mejorada del optimizador **Adam** (Adaptive Moment Estimation), que incluye **Weight Decay** (decaimiento de peso) de forma correcta.

### Componentes de AdamW:

1. **Momentum (primer momento)**:
   - Mantiene un promedio de los gradientes pasados
   - Ayuda a continuar en la dirección correcta
   - Imagina una bola rodando por una colina

2. **RMSProp (segundo momento)**:
   - Mantiene un promedio de los gradientes al cuadrado
   - Adapta el learning rate por parámetro
   - Parámetros con gradientes grandes → learning rate pequeño
   - Parámetros con gradientes pequeños → learning rate grande

3. **Weight Decay (decaimiento de peso)**:
   - Multiplica cada peso por un factor < 1 (ej: 0.99)
   - Penaliza pesos grandes
   - Evita overfitting
   - Regularización L2

### ¿Por qué AdamW y no SGD o Adam clásico?

| Optimizador | Ventajas | Desventajas | ¿Por qué no? |
|-------------|----------|--------------|---------------|
| **AdamW** | Adapta LR, incluye regularización | Más memoria | ✅ Elegido |
| SGD | Simple, estable | LR fijo, necesita tuning | Más epochs |
| Adam | Adam sin weight decay correcto | Weight decay aplicado incorrectamente | Menos preciso |

### ¿Por qué learning rate de 2e-5 (0.00002)?

- **Valor del paper original de BERT**: Los autores probaron diferentes valores y encontraron 2e-5 óptimo
- **Bajo LR para modelos pre-entrenados**: Con transfer learning, no queremos cambiar demasiado los pesos ya aprendidos
- **Si es muy alto**: El modelo diverge (explota gradientes)
- **Si es muy bajo**: El modelo no aprende (subfitting)

### Código interno:
```python
# Pseudocódigo de AdamW
m = beta1 * m + (1 - beta1) * gradiente          # Momentum
v = beta2 * v + (1 - beta2) * gradiente**2       # RMSProp
peso = peso - lr * m / (sqrt(v) + epsilon)      # Actualizar
peso = peso * (1 - lr * weight_decay)           # Weight decay
```

## 5.4 Scheduler (Learning Rate con Warmup)

```python
# Learning rate lineal con warmup del 10%
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)
```

### ¿Qué es un Scheduler?

Un **scheduler** (programador) ajusta el learning rate durante el entrenamiento. En lugar de mantener un LR fijo, lo va cambiando según un cronograma.

### ¿Por qué necesitamos warmup?

El **warmup** significa comenzar con un learning rate muy bajo e ir aumentándolo gradualmente.

**Problema sin warmup:**
- Al inicio, los pesos están randomizados
- Un LR alto al principio causa actualizaciones grandes e inestables
- El modelo puede diverge o oscilar

**Con warmup:**
- LR comienza en 0 (o muy bajo)
- Aumenta gradualmente hasta alcanzar el LR objetivo
- Da tiempo al modelo de "estabilizarse"
- Luego puede reducirse para converger mejor

### Tipos de Scheduler:

| Tipo | Descripción | Uso común |
|------|-------------|-----------|
| **Linear warmup** | LR aumenta linealmente | BERT, Transformers |
| **Cosine annealing** | LR sigue curva coseno | Transformers grandes |
| **Step decay** | LR reduce en pasos específicos | CNNs clásicas |
| **Reduce on plateau** | Reduce cuando metricas se estancan | General |

### Nuestro Scheduler (Linear Warmup):

```python
# En el entrenamiento:
for step in range(total_steps):
    # Al principio: LR = 0 (o lr * step/warmup_steps)
    # En el medio: LR = lr_base
    # Al final: LR = lr_base * (1 - step/total_steps)
    optimizer.step()
    scheduler.step()
```

### ¿Por qué usamos warmup del 10%?

- **Estabilidad inicial**: evita actualizaciones grandes cuando los pesos de la capa clasificadora aún son aleatorios.
- **Convergencia suave**: tras el warmup, el LR decae linealmente hasta el final.
- **Costo mínimo**: con 3 épocas, el warmup solo cubre una fracción pequeña de los pasos.

### Gráfico del Learning Rate:

```
Sin scheduler (constante):
LR: ─────────────────────────────

Con scheduler (linear decay):
LR: ╲___________________________
    ↑                          ↑
   inicio                    final
```

### ¿Por qué reducir el LR al final?

- Al acercarse al óptimo, quieres movimientos más precisos
- LR alto te hace "saltar" sobre el mínimo
- LR bajo te permite converger suavemente

---

## 5.5 Ejemplo Completo del Loop de Entrenamiento

```python
# Pseudocódigo del entrenamiento
for epoch in range(EPOCHS):
    for batch in train_loader:
        # 1. Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 2. Backward pass
        loss.backward()
        
        # 3. Clip gradientes (evitar explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 4. Actualizar pesos (AdamW)
        optimizer.step()
        
        # 5. Actualizar LR (Scheduler)
        scheduler.step()
        
        # 6. Limpiar gradientes
        optimizer.zero_grad()
```

### Resumen del flujo:
1. **Forward**: El texto pasa por BERT → predicción
2. **Loss**: Se calcula qué tan mala fue la predicción
3. **Backward**: Se calculan los gradientes
4. **Clip**: Se limitan los gradientes para evitar errores numéricos
5. **Optimizer step**: Se actualizan los pesos
6. **Scheduler step**: Se ajusta el learning rate
7. **Zero grad**: Se limpian gradientes para el siguiente batch
- Luego aumenta gradualmente
- Al final, disminuye para converger mejor

---

# 6. DATASET DE ENTRENAMIENTO

## 6.1 Características

| Métrica | Valor |
|---------|-------|
| Total de muestras | 10,000 |
| Número de categorías | 42 |
| División train/validation | 80%/20% |
| Muestras entrenamiento | ~8,000 |
| Muestras validación | ~2,000 |

## 6.2 Distribución de Categorías

```
Technology                      2511 (25.1%)
Data & Analytics                 568 (5.7%)
Healthcare                       488 (4.9%)
Marketing & Sales                463 (4.6%)
Engineering & Manufacturing      435 (4.4%)
... (37 más categorías)
```

**Problema**: Technology tiene muchos más ejemplos que otras categorías.

**Solución**: Usamos `stratify` en el split para mantener proporciones.

## 6.3 Preprocesamiento del Texto

```python
# Pipeline de limpieza
1. Normalizar Unicode (eliminar acentos)
2. Expandir contracciones (don't → do not)
3. Eliminar caracteres especiales
4. Eliminar números
5. Normalizar espacios
6. Convertir a minúsculas

# Resultado:
# Input: "I'm a Software Engineer! 🎯"
# Output: "i am a software engineer"
```

---

# 7. MÉTRICAS Y RENDIMIENTO

## 7.1 Métricas del Modelo

El entrenamiento escribe `red_neuronal/models/metrics.json` con:

- `accuracy`, `macro_f1`, `weighted_f1`
- `classification_report` y `confusion_matrix`
- `accuracy_train` y `accuracy_val` por época
- hiperparámetros y número de muestras

La página `/modelo/` lee ese archivo. Si no existe, muestra valores ilustrativos y lo indica explícitamente.

```python
# Formato de las series por época en metrics.json (valores de ejemplo)
Epoch 1 - Validation Accuracy: 0.8234
Epoch 2 - Validation Accuracy: 0.8712
Epoch 3 - Validation Accuracy: 0.8934
```

## 7.2 Parámetros del Modelo

| Parámetro | Valor |
|----------|-------|
| Parámetros totales | ~110,000,000 |
| Parámetros entrenables | ~110,000,000 |
| Capas transformer | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Vocabulario | 30,522 tokens |

## 7.3 Limitaciones Conocidas

1. **Desbalance de clases**: Technology domina el dataset; se mitiga con `class_weights="balanced"` y split estratificado.
2. **512 tokens de límite**: Currículums muy largos se truncan.
3. **Solo inglés**: El modelo está en inglés; se traduce antes de predecir.
4. **Traducción imperfecta**: La traducción automática puede perder matiz.
5. **Dependencia del mapeo**: `label_mapping.json` debe corresponder al checkpoint; si se reconstruye desde el CSV, el orden de clases puede cambiar.
6. **Modelo ausente**: la app falla de forma controlada en lugar de predecir con pesos aleatorios.

---

# 8. FLUJO DE PREDICCIÓN (INFERENCIA)

## 8.1 Pipeline Completo

```
Usuario sube PDF
    ↓
[1] Extraer texto (híbrido)
    ├── PyMuPDF page.get_text() (texto embebido)
    └── Tesseract OCR solo si la página no tiene texto suficiente
    ↓
[2] Detectar idioma y traducir
    └── langdetect + GoogleTranslator
    ↓
[3] Procesamiento NLP
    ├── spaCy NER (detectar entidades)
    ├── Extracción de secciones
    └── Detección de tecnologías
    ↓
[4] Enriquecer texto
    ├── Experience + Education + Skills
    ├── Keywords detectados
    └── Tecnologías identificadas
    ↓
[5] Limpiar texto
    └── TextCleaner (pipeline de limpieza)
    ↓
[6] Tokenizar con BERT
    └── 512 tokens máximo
    ↓
[7] Modelo BERT
    └── Forward pass → probabilidades
    ↓
[8] Predicción
    └── Argmax → categoría + confianza
    ↓
Mostrar resultado
```

## 8.2 ¿Por qué todo este proceso?

Cada paso es necesario:

| Paso | ¿Por qué? |
|------|-----------|
| OCR | PDFs escaneados no tienen texto digital |
| Traducción | El modelo fue entrenado en inglés |
| NER | Detectar entidades da contexto adicional |
| Secciones | Experience/Skills son los más relevantes |
| Limpiar | Elimina ruido que confunde al modelo |
| 512 tokens | Límite de BERT (arquitectura) |

---

# 9. RESUMEN TÉCNICO PARA EXÁMEN

## Preguntas Probables del Profesor:

### 1. ¿Por qué usar BERT y no otra arquitectura?
> BERT es el estado del arte en clasificación de texto. Su arquitectura bidirectional pre-entrenada captura contexto completo, y el transfer learning permite adaptarlo con pocos datos.

### 2. ¿Por qué PyTorch?
> PyTorch ofrece la mejor combinación de flexibilidad, debuggeo y ecosistema. La integración con transformers de Hugging Face es seamless.

### 3. ¿Por qué 512 tokens?
> Es el límite de arquitectura de BERT. 512 tokens son suficientes para un CV típico (~2000-3000 palabras).

### 4. ¿Por qué 3 épocas?
> Es un balance entre tiempo de entrenamiento y precisión. Además hay early stopping: si la validación deja de mejorar durante 2 épocas, se detiene y se conserva el mejor checkpoint.

### 5. ¿Por qué Tesseract para OCR?
> Tesseract es open source, no tiene costo, y ofrece precisión aceptable. Alternativas como Textract son de pago. Además, solo se usa como fallback cuando el PDF no trae texto embebido.

### 6. ¿Cómo se maneja el desbalance de clases?
> Usamos stratified split para mantener proporciones y `class_weights="balanced"` en la pérdida para que las clases minoritarias tengan más peso.

### 7. ¿Qué es transfer learning?
> Es usar un modelo pre-entrenado (BERT en este caso) y ajustarlo a una tarea específica. No entrenamos desde cero.

### 8. ¿Por qué dropout de 30%?
> Es un valor empírico estándar para fine-tuning de BERT. Suficiente para evitar overfitting sin perder capacidad.

---

# 10. REFERENCIAS Y CRÉDITOS

- **BERT Paper**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
- **PyTorch**: https://pytorch.org
- **Hugging Face Transformers**: https://huggingface.co/transformers
- **spaCy**: https://spacy.io
- **Tesseract**: https://github.com/tesseract-ocr/tesseract

---

*Documento creado para presentación del proyecto de clasificación de currículums con BERT*