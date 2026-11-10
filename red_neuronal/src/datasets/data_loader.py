"""
=================================================================
CARGADOR DE DATASETS
Universidad - Carga de Datos para Entrenamiento

Este módulo carga el dataset principal para entrenar el modelo 
BERT de clasificación de currículums.

Dataset necesario:
- training_data.csv: 10,000 currículums categorizados

Dependencias:
- pandas: lectura y manipulación de CSVs
- os: verificación de existencia de archivos

=================================================================
"""

import pandas as pd           # Biblioteca para manejar datos tabulares (CSV, Excel)
import os                     # Operaciones del sistema de archivos
from typing import Dict, Optional  # Type hints para mejor documentación
from ..utils.config import DATASET_PATHS  # Configuración de rutas


class DatasetLoader:
    """
    Cargador del dataset de entrenamiento para BERT.
    
    Este loader está diseñado específicamente para cargar los datos
    necesarios para entrenar el clasificador de currículums.
    
    Atributos:
        training_df: DataFrame con datos de entrenamiento
    
    Ejemplo de uso:
        loader = DatasetLoader()
        df = loader.load_training_data()
        texts = df['Resume Text'].tolist()
        labels = df['Category'].tolist()
    """
    
    def __init__(self):
        """
        Inicializa el Cargador de Datasets.
        """
        self.training_df = None

    def load_training_data(self) -> pd.DataFrame:
        """
        Carga el dataset de entrenamiento para BERT.
        
        El dataset training_data.csv contiene:
        - 10,000 currículums categorizados
        - 42 categorías profesionales
        - Columnas: Resume Text, Category
        
        Returns:
            pd.DataFrame: Dataset de entrenamiento
            
        Raises:
            FileNotFoundError: Si no existe training_data.csv
        """
        print("Loading training data...")
        
        # Ruta del dataset principal
        training_path = DATASET_PATHS["1_resume_classification_training"]
        
        if not os.path.exists(training_path):
            raise FileNotFoundError(f"Dataset no encontrado: {training_path}")
        
        # Cargar datos
        self.training_df = pd.read_csv(training_path)
        print(f"  - training_data.csv: {len(self.training_df)} rows, {len(self.training_df.columns)} columns")
        print(f"  - Categorías únicas: {self.training_df['Category'].nunique()}")
        
        return self.training_df

    def get_training_data(self) -> pd.DataFrame:
        """
        Obtiene datos de entrenamiento.
        
        Returns:
            DataFrame con datos de entrenamiento
            
        Raises:
            ValueError: Si no se han cargado los datos
        """
        if self.training_df is not None:
            return self.training_df
        raise ValueError("No training data loaded. Call load_training_data() first.")

    def get_texts_and_labels(self) -> tuple:
        """
        Obtiene textos y etiquetas separados para entrenamiento.
        
        Returns:
            tuple: (texts list, labels list)
            
        Raises:
            ValueError: Si no se han cargado los datos
        """
        if self.training_df is None:
            raise ValueError("No training data loaded. Call load_training_data() first.")
        
        # Detectar nombres de columnas (el dataset actual usa "Resume Text").
        text_candidates = [
            "Resume Text",
            "Resume_str",
            "Resume",
            "resume_text",
            "text",
        ]
        text_col = next(
            (col for col in text_candidates if col in self.training_df.columns),
            None,
        )
        if text_col is None:
            raise ValueError(
                "No se encontró una columna de texto. "
                f"Columnas disponibles: {list(self.training_df.columns)}"
            )

        label_col = (
            "Category"
            if "Category" in self.training_df.columns
            else self.training_df.columns[-1]
        )

        texts = self.training_df[text_col].tolist()
        labels = self.training_df[label_col].tolist()

        return texts, labels