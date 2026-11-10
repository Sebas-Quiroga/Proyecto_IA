"""
Sistema Inteligente de Análisis de Hojas de Vida
Clasificación de currículums usando BERT
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.data_loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.models.bert_classifier import BertClassifierModel
from src.preprocessing.text_cleaner import TextCleaner
from src.utils.config import LABEL_MAPPING_PATH, MODEL_PATH

class ResumeAnalysisSystem:
    def __init__(self, use_gpu: bool = True):
        self.device = "cuda" if use_gpu and os.name != "nt" else "cpu"
        print(f"Initializing system on device: {self.device}")

        self.text_cleaner = TextCleaner()
        self.classifier = None
        self.preprocessor = None
        self.label_mapping = None

    def initialize(self):
        print("\n=== Loading and Preparing Data ===")
        self.preprocessor = DatasetPreprocessor()
        print("\n=== System Ready ===")

    def train_classifier(self, epochs: int = 3, batch_size: int = 16):
        print("\n=== Training Classifier ===")
        loader = DatasetLoader()
        df = loader.load_training_data()

        texts, labels = self.preprocessor.prepare_classification_data(
            df, "Resume Text", "Category"
        )
        train_texts, val_texts, train_labels, val_labels = (
            self.preprocessor.prepare_for_bert(texts, labels)
        )

        self.label_mapping, self.reverse_mapping = (
            self.preprocessor.get_label_mapping()
        )
        num_classes = len(self.label_mapping)

        print(f"\nTraining with {num_classes} classes...")
        self.classifier = BertClassifierModel(
            num_classes=num_classes, device=self.device
        )
        self.classifier.train(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            epochs=epochs,
            batch_size=batch_size,
            save_path=str(MODEL_PATH),
        )
        self.preprocessor.save_label_mapping(str(LABEL_MAPPING_PATH))

    def analyze_resume(self, resume_text: str):
        print("\n=== Analyzing Resume ===")

        cleaned_text = self.text_cleaner.clean(resume_text)
        print(f"1. Text cleaned: {len(cleaned_text)} characters")

        predicted_category = None
        if self.classifier:
            prediction = self.classifier.predict([cleaned_text])[0]
            predicted_category = self.reverse_mapping.get(prediction, "Unknown")
            print(f"2. Predicted Category: {predicted_category}")

        return {
            "cleaned_text": cleaned_text,
            "predicted_category": predicted_category
        }

def main():
    system = ResumeAnalysisSystem(use_gpu=False)
    system.initialize()

    sample_resume = """
    EXPERIENCE
    Software Engineer at Tech Corp
    - Developed Python applications using Django and Flask
    - Worked with PostgreSQL, MongoDB, and Redis
    - Deployed applications using AWS and Docker
    - Collaborated with team using Git and CI/CD pipelines

    SKILLS
    Python, JavaScript, Django, Flask, React, AWS, Docker, PostgreSQL, Git

    EDUCATION
    BS Computer Science - University of Technology
    """

    result = system.analyze_resume(sample_resume)
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()