"""
Script de humo para comprobar que el pipeline de entrenamiento funciona.

No es un test de pytest: entrena solo 100 muestras para validar imports,
datos, modelo y guardado.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.data_loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.models.bert_classifier import BertClassifierModel
from src.utils.config import MODEL_DIR


def main():
    print("1. Cargando dataset de entrenamiento...")
    loader = DatasetLoader()
    df = loader.load_training_data()
    print(f"   Filas: {len(df)}, Columnas: {len(df.columns)}")

    print("\n2. Preparando datos...")
    preprocessor = DatasetPreprocessor()
    texts, labels = preprocessor.prepare_classification_data(
        df, "Resume Text", "Category"
    )
    train_texts, val_texts, train_labels, val_labels = (
        preprocessor.prepare_for_bert(texts, labels, val_size=0.2)
    )
    label_mapping, reverse_mapping = preprocessor.get_label_mapping()
    print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}")
    print(f"   Categorías: {len(label_mapping)}")

    print("\n3. Entrenando clasificador (1 época, 100 muestras)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    classifier = BertClassifierModel(
        num_classes=len(label_mapping), device="cpu"
    )
    classifier.train(
        train_texts[:100],
        train_labels[:100],
        val_texts[:20],
        val_labels[:20],
        epochs=1,
        batch_size=8,
        save_path=str(MODEL_DIR / "test_classifier.pt"),
    )
    print("   Entrenamiento completado!")

    print("\n4. Predicción de ejemplo...")
    test_text = "Python developer with Django and PostgreSQL experience"
    prediction = classifier.predict([test_text])
    print(f"   Predicción: {reverse_mapping[prediction[0]]}")

    print("\n✅ Todo funcionando!")


if __name__ == "__main__":
    main()
