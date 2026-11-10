"""
Entrenamiento del clasificador BERT de currículums.

Usa el mismo pipeline de datos y la misma arquitectura que la aplicación web
para evitar diferencias entre entrenamiento e inferencia.
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.data_loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.models.bert_classifier import BertClassifierModel
from src.utils.config import (
    BATCH_SIZE,
    BERT_MODEL,
    EPOCHS,
    LABEL_MAPPING_PATH,
    LEARNING_RATE,
    MAX_LEN,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUM_WORKERS,
    RANDOM_SEED,
)


def main():
    loader = DatasetLoader()
    df = loader.load_training_data()

    preprocessor = DatasetPreprocessor()
    texts, labels = preprocessor.prepare_classification_data(
        df, text_col="Resume Text", label_col="Category"
    )
    train_texts, val_texts, train_labels, val_labels = (
        preprocessor.prepare_for_bert(texts, labels, val_size=0.2)
    )
    label_mapping, reverse_mapping = preprocessor.get_label_mapping()
    num_classes = len(label_mapping)

    # Pesos por clase para compensar el desbalance del dataset.
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(train_labels),
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    classifier = BertClassifierModel(
        num_classes=num_classes,
        model_name=BERT_MODEL,
        device="cuda",
    )
    class_weights = class_weights.to(classifier.device)
    best_accuracy = classifier.train(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_len=MAX_LEN,
        save_path=str(MODEL_PATH),
        class_weights=class_weights,
        num_workers=NUM_WORKERS,
    )

    # El mapeo se guarda junto al modelo: la inferencia no debe reconstruirlo
    # desde el CSV porque el orden podría cambiar.
    preprocessor.save_label_mapping(str(LABEL_MAPPING_PATH))

    # Métricas reales para la página /modelo/.
    predictions = classifier.predict(val_texts, max_len=MAX_LEN)
    metrics = {
        "accuracy": float(accuracy_score(val_labels, predictions)),
        "macro_f1": float(
            f1_score(val_labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(val_labels, predictions, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            val_labels,
            predictions,
            target_names=[
                reverse_mapping[index] for index in range(num_classes)
            ],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            val_labels, predictions
        ).tolist(),
        "best_val_accuracy": float(best_accuracy),
        # Series por época para la gráfica de /modelo/ (en porcentaje).
        "accuracy_train": [
            round(value * 100, 2)
            for value in classifier.history.get("train_accuracy", [])
        ],
        "accuracy_val": [
            round(value * 100, 2)
            for value in classifier.history.get("val_accuracy", [])
        ],
        "train_loss": [
            round(value, 6)
            for value in classifier.history.get("train_loss", [])
        ],
        "num_classes": num_classes,
        "num_train_samples": int(len(train_texts)),
        "num_val_samples": int(len(val_texts)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_len": MAX_LEN,
        "random_seed": RANDOM_SEED,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Mapeo de etiquetas: {LABEL_MAPPING_PATH}")
    print(f"Métricas: {METRICS_PATH}")
    print(f"Accuracy validación: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
