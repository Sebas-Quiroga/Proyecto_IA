from src.utils import config


def test_model_dir_is_next_to_red_neuronal():
    assert config.MODEL_DIR.name == "models"
    assert config.MODEL_DIR.parent.name == "red_neuronal"


def test_dataset_path_exists():
    path = config.DATASET_PATHS["1_resume_classification_training"]
    assert path.endswith("training_data.csv")
