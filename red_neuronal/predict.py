"""
Predicción de categoría de CV usando el modelo entrenado.

Uso:
    python predict.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.preprocessor import DatasetPreprocessor
from src.models.bert_classifier import BertClassifierModel
from src.preprocessing.text_cleaner import TextCleaner
from src.utils.config import LABEL_MAPPING_PATH, MODEL_PATH


def load_classifier_and_mapping():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo en {MODEL_PATH}. "
            "Ejecuta train_bert_classifier.py primero."
        )

    if LABEL_MAPPING_PATH.exists():
        label_mapping, reverse_mapping = DatasetPreprocessor.load_label_mapping(
            str(LABEL_MAPPING_PATH)
        )
    else:
        # Compatibilidad con modelos antiguos que no guardaban el mapeo.
        from src.datasets.data_loader import DatasetLoader

        loader = DatasetLoader()
        df = loader.load_training_data()
        preprocessor = DatasetPreprocessor()
        preprocessor.prepare_classification_data(
            df, "Resume Text", "Category"
        )
        label_mapping, reverse_mapping = preprocessor.get_label_mapping()

    classifier = BertClassifierModel(
        num_classes=len(label_mapping), device="cpu"
    )
    classifier.load_model(str(MODEL_PATH))
    return classifier, reverse_mapping


def predict(cv_text: str):
    classifier, reverse_mapping = load_classifier_and_mapping()
    cleaned = TextCleaner().clean(cv_text)
    probs = classifier.predict_proba([cleaned])[0]

    results = [
        {
            "category": reverse_mapping.get(index, f"Unknown_{index}"),
            "probability": float(prob * 100),
        }
        for index, prob in enumerate(probs)
    ]
    results.sort(key=lambda item: item["probability"], reverse=True)

    print("\n" + "=" * 60)
    print("PREDICCIÓN DE CATEGORÍA PROFESIONAL")
    print("=" * 60)
    print(f"\n🎯 Categoría: {results[0]['category']}")
    print(f"📊 Confianza: {results[0]['probability']:.2f}%\n")
    print("TOP 5 CATEGORÍAS:")
    print("-" * 40)
    for index, result in enumerate(results[:5], start=1):
        print(f"{index}. {result['category']:35} {result['probability']:.2f}%")

    return results


if __name__ == "__main__":
    cv_example = """
    EXPERIENCE
    Software Engineer at Tech Corp
    - Developed Python applications using Django and Flask
    - Worked with PostgreSQL, MongoDB, and Redis
    - Deployed applications using AWS and Docker

    SKILLS
    Python, JavaScript, Django, AWS, Docker, PostgreSQL

    EDUCATION
    BS Computer Science
    """
    predict(cv_example)
