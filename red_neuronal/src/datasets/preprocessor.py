"""
=================================================================
PREPROCESADOR DE DATASET PARA BERT
Universidad - Preparación de Datos para Entrenamiento

Este módulo preprocesa los datos del dataset antes de entrenar
el modelo BERT:

Pipeline de preprocesamiento:
1. Limpieza de datos (valores nulos, vacíos)
2. Limpieza de texto (TextCleaner)
3. Eliminación de duplicados
4. Codificación de etiquetas (categorías -> números)
5. Split train/validation (80/20)

Dependencias:
- pandas: manipulación de datos
- sklearn: train_test_split para splitting
- TextCleaner: limpieza de texto

=================================================================
"""

import json                  # Persistencia del mapeo de etiquetas
import os
import pandas as pd          # Manipulación de datos tabulares
import numpy as np           # Operaciones numéricas
from typing import Tuple, Optional  # Type hints
from ..preprocessing.text_cleaner import TextCleaner  # Limpiador de texto


class DatasetPreprocessor:
    """
    Preprocesador de datos para entrenamiento de BERT.
    
    Maneja todo el pipeline de preparación de datos:
    - Limpieza de valores nulos y vacíos
    - Limpieza de texto
    - Eliminación de duplicados
    - Codificación de etiquetas
    - Split entrenamiento/validación
    
    Atributos:
        text_cleaner: instancia de TextCleaner
        label_mapping: {categoría: índice}
        reverse_label_mapping: {índice: categoría}
    
    Ejemplo de uso:
        preprocessor = DatasetPreprocessor()
        texts, labels = preprocessor.prepare_classification_data(df, "Resume Text", "Category")
        train_texts, val_texts, train_labels, val_labels = preprocessor.prepare_for_bert(texts, labels)
    """
    
    def __init__(self):
        """
        Inicializa el preprocesador.
        
        Crea instancia de TextCleaner para limpiar textos.
        """
        self.text_cleaner = TextCleaner()
        self.label_encoder = None

    def prepare_classification_data(self, df: pd.DataFrame, text_col: str = "Resume Text", label_col: str = "Category") -> Tuple[pd.Series, pd.Series]:
        """
        Prepara datos para clasificación.
        
        Pipeline completo:
        1. Eliminar filas con valores nulos en columna de texto
        2. Eliminar filas con texto vacío
        3. Eliminar textos muy cortos (<10 caracteres)
        4. Limpiar texto con TextCleaner
        5. Eliminar textos muy cortos después de limpieza
        6. Eliminar duplicados
        7. Codificar etiquetas a números
        
        Args:
            df: DataFrame con los datos
            text_col: nombre de la columna con el texto
            label_col: nombre de la columna con la categoría
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (textos, etiquetas)
        
        Dependencias:
            - df.dropna: eliminar valores nulos
            - df.drop_duplicates: eliminar duplicados
            - TextCleaner.clean: limpiar texto
        """
        print("Preparing classification data...")

        if text_col not in df.columns:
            raise ValueError(
                f"Columna de texto '{text_col}' no encontrada. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        if label_col not in df.columns:
            raise ValueError(
                f"Columna de etiquetas '{label_col}' no encontrada. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Copiar para no modificar el original
        df = df.copy()

        # ========== LIMPIEZA DE DATOS ==========
        
        # Eliminar filas con valores nulos en la columna de texto
        df = df.dropna(subset=[text_col])
        
        # Eliminar filas donde el texto sea vacío o solo espacios
        df = df[df[text_col].astype(str).str.strip() != ""]
        
        # Eliminar textos muy cortos (menos de 10 caracteres)
        df = df[df[text_col].str.len() > 10]

        print(f"After dropping nulls/empty: {len(df)} samples")

        # ========== LIMPIEZA DE TEXTO ==========
        
        # Aplicar TextCleaner a cada texto
        #清洗 = clean(text) -> lowercase, remove special chars, etc.
        df['cleaned_text'] = df[text_col].apply(self.text_cleaner.clean)
        
        # Eliminar textos que quedaron muy cortos después de limpieza
        df = df.dropna(subset=['cleaned_text'])
        df = df[df['cleaned_text'].str.len() > 10]

        print(f"After cleaning: {len(df)} samples")

        # ========== ELIMINAR DUPLICADOS ==========
        
        # Eliminar textos duplicados (mismo contenido)
        df = df.drop_duplicates(subset=['cleaned_text'])
        print(f"After dedup: {len(df)} samples")

        # ========== CODIFICAR ETIQUETAS ==========
        
        self._encode_labels(df, label_col)
        
        # Retornar textos limpios y etiquetas codificadas
        return df['cleaned_text'], df['label']

    def _encode_labels(self, df: pd.DataFrame, label_col: str):
        """
        Codifica las categorías a números.
        
        Proceso:
        1. Obtener todas las etiquetas únicas
        2. Crear mapeo: {categoría: índice}
        3. Crear mapeo inverso: {índice: categoría}
        4. Agregar columna 'label' con los índices
        
        Args:
            df: DataFrame con los datos
            label_col: nombre de la columna de categorías
        
        Resultado:
            - label_mapping: {"Technology": 0, "HR": 1, ...}
            - reverse_label_mapping: {0: "Technology", 1: "HR", ...}
        
        Ejemplo:
            Input: ["Technology", "HR", "Technology"]
            Output: [0, 1, 0]
        """
        unique_labels = df[label_col].unique()
        
        # Crear mapeo: categoría -> índice numérico
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Crear mapeo inverso: índice -> categoría
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        # Agregar columna numérica
        df['label'] = df[label_col].map(self.label_mapping)
        print(f"Encoded {len(unique_labels)} unique labels")

    def prepare_for_bert(self, texts: pd.Series, labels: pd.Series, val_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Divide los datos en entrenamiento y validación.
        
        Usa stratified split para mantener la proporción de cada categoría.
        
        Args:
            texts: Serie con textos
            labels: Serie con etiquetas (números)
            val_size: proporción para validación (default: 0.2 = 20%)
            random_state: semilla para reproducibilidad
        
        Returns:
            Tuple: (train_texts, val_texts, train_labels, val_labels)
        
        Proceso:
            - 80% entrenamiento
            - 20% validación
            - Mismas proporciones de categorías en ambos sets
        
        Dependencias:
            - sklearn.model_selection.train_test_split: división de datos
            - stratify: mantener proporción de clases
        """
        from sklearn.model_selection import train_test_split
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts.values, labels.values,
            test_size=val_size,
            random_state=random_state,
            stratify=labels.values  # Mantener proporción de categorías
        )
        
        print(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")
        return train_texts, val_texts, train_labels, val_labels

    def get_label_mapping(self):
        """
        Obtiene los mapeos de etiquetas.

        Returns:
            Tuple: (label_mapping, reverse_label_mapping)
        """
        return self.label_mapping, self.reverse_label_mapping

    def save_label_mapping(self, path: str) -> None:
        """Guarda ``{categoría: índice}`` en JSON para usarlo en inferencia."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                self.label_mapping,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    @staticmethod
    def load_label_mapping(path: str):
        """Carga el mapeo de etiquetas desde JSON."""
        with open(path, "r", encoding="utf-8") as file:
            label_mapping = json.load(file)
        label_mapping = {
            str(category): int(index)
            for category, index in label_mapping.items()
        }
        reverse_label_mapping = {
            index: category for category, index in label_mapping.items()
        }
        return label_mapping, reverse_label_mapping