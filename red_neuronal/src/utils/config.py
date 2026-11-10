"""
Configuración central del proyecto.

Todas las rutas se calculan desde la ubicación de este archivo para que el
proyecto funcione igual desde la raíz, desde ``red_neuronal`` o desde Django.
"""

import os
from pathlib import Path


# ====================================================================
# RUTAS BASE
# ====================================================================

# .../red_neuronal/src/utils/config.py -> .../red_neuronal/src
SRC_DIR = Path(__file__).resolve().parent.parent
# .../red_neuronal
RED_NEURONAL_DIR = SRC_DIR.parent
# raíz del repositorio
PROJECT_DIR = RED_NEURONAL_DIR.parent

DATASET_DIR = PROJECT_DIR / "datos_entrenamiento"
MODEL_DIR = RED_NEURONAL_DIR / "models"
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", MODEL_DIR / "bert_classifier_category.pt")
)
LABEL_MAPPING_PATH = Path(
    os.environ.get("LABEL_MAPPING_PATH", MODEL_DIR / "label_mapping.json")
)
METRICS_PATH = Path(os.environ.get("METRICS_PATH", MODEL_DIR / "metrics.json"))


# ====================================================================
# DATASETS
# ====================================================================

DATASET_PATHS = {
    "1_resume_classification_training": str(
        DATASET_DIR / "1_resume_classification" / "training_data.csv"
    ),
    "1_resume_classification_jobs": str(
        DATASET_DIR / "1_resume_classification" / "job_roles.csv"
    ),
    "1_resume_classification_skills": str(
        DATASET_DIR / "1_resume_classification" / "skills_list.csv"
    ),
    "2_real_pdf_resumes": str(
        DATASET_DIR / "2_real_pdf_resumes" / "Resume" / "Resume.csv"
    ),
    "3_job_matching": str(
        DATASET_DIR / "3_job_resume_matching" / "job_resume_fit.csv"
    ),
    "5_skills": str(DATASET_DIR / "5_skills_taxonomy" / "skills_en.csv"),
    "5_occupations": str(
        DATASET_DIR / "5_skills_taxonomy" / "occupations_en.csv"
    ),
    "5_skill_relations": str(
        DATASET_DIR / "5_skills_taxonomy" / "occupationSkillRelations.csv"
    ),
}


# ====================================================================
# PARÁMETROS DE ENTRENAMIENTO
# ====================================================================

MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_WORKERS = 0  # 0 es más seguro en Windows y en contenedores pequeños
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 2

# Modelo pre-entrenado
BERT_MODEL = os.environ.get("BERT_MODEL", "bert-base-uncased")

# Valores por defecto para mostrar en la web sin cargar el modelo completo.
BERT_CONFIG = {
    "total_params": 110_000_000,
    "num_layers": 12,
    "hidden_size": 768,
    "num_heads": 12,
    "vocab_size": 30_522,
}

# Dispositivo solicitado; BertClassifierModel hace el fallback a CPU.
DEVICE = "cuda"
