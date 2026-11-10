"""
Vistas Django del analizador de currículums.

La lógica pesada (PyTorch, PyMuPDF, Tesseract, spaCy, traducción) se importa
dentro de las funciones para no cargarla al arrancar el proceso.
"""

import hashlib
import io
import json
import logging
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import redirect, render

logger = logging.getLogger(__name__)

# ====================================================================
# ESTADO GLOBAL (singletons con lock)
# ====================================================================

_classifier = None
_classifier_lock = threading.Lock()

_label_mapping = None
_reverse_mapping = None
_mapping_lock = threading.Lock()

_text_cleaner = None
_skill_extractor = None

DEFAULT_CATEGORY_COUNTS = {
    "Technology": 2511,
    "Data & Analytics": 568,
    "Healthcare": 488,
    "Marketing & Sales": 463,
    "Engineering & Manufacturing": 435,
    "Finance": 380,
    "Human Resources": 320,
    "Design": 280,
    "Education": 250,
    "Operations": 200,
    "Consulting": 180,
    "Legal": 150,
    "Research": 120,
    "Product": 100,
    "Quality": 80,
}

DEFAULT_METRICS = {
    "accuracy_train": [82.34, 87.12, 89.34],
    "accuracy_val": [81.00, 86.00, 89.00],
    "illustrative": True,
}


# ====================================================================
# HELPERS
# ====================================================================

def _no_cache(response):
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


def get_text_cleaner():
    global _text_cleaner
    if _text_cleaner is None:
        from src.preprocessing.text_cleaner import TextCleaner

        _text_cleaner = TextCleaner()
    return _text_cleaner


def get_skill_extractor():
    global _skill_extractor
    if _skill_extractor is None:
        from src.preprocessing.skill_extractor import SkillExtractor

        _skill_extractor = SkillExtractor()
    return _skill_extractor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ====================================================================
# MODELO Y ETIQUETAS
# ====================================================================

def ensure_model_exists() -> bool:
    """
    Verifica que el modelo exista y sea válido; si no, lo descarga.

    La descarga se hace a un archivo temporal y se reemplaza de forma atómica
    para no dejar un .pt parcial que luego parezca válido.
    """
    try:
        import gdown
    except ImportError:
        logger.error(
            "gdown no está instalado; no se puede descargar el modelo. "
            "Instala las dependencias o coloca el modelo en MODEL_PATH."
        )
        return False

    model_path = Path(settings.MODEL_PATH)
    model_dir = model_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)

    def _valid(path: Path) -> bool:
        if not path.exists() or path.stat().st_size < settings.MODEL_MIN_SIZE:
            return False
        if settings.MODEL_SHA256:
            actual = _sha256(path)
            if actual.lower() != settings.MODEL_SHA256.lower():
                logger.error(
                    "Checksum inválido para %s: %s != %s",
                    path,
                    actual,
                    settings.MODEL_SHA256,
                )
                return False
        return True

    if _valid(model_path):
        logger.info("Modelo disponible: %s", model_path)
        return True

    if model_path.exists():
        logger.warning("Modelo incompleto o inválido; se volverá a descargar.")
        try:
            model_path.unlink()
        except OSError:
            logger.exception("No se pudo eliminar el modelo inválido.")

    if not settings.MODEL_GOOGLE_DRIVE_ID:
        logger.error("MODEL_GOOGLE_DRIVE_ID no está configurado.")
        return False

    temp_path = model_path.with_suffix(model_path.suffix + ".part")
    url = f"https://drive.google.com/uc?id={settings.MODEL_GOOGLE_DRIVE_ID}"

    try:
        logger.info("Descargando modelo desde Google Drive...")
        gdown.download(url, str(temp_path), quiet=False)
        if not _valid(temp_path):
            logger.error("La descarga del modelo no es válida.")
            temp_path.unlink(missing_ok=True)
            return False
        os.replace(temp_path, model_path)
        logger.info("Modelo descargado en %s", model_path)
        return True
    except Exception:
        logger.exception("Error descargando el modelo.")
        temp_path.unlink(missing_ok=True)
        return False


def load_mappings():
    """
    Carga ``{categoría: índice}`` desde el JSON guardado con el modelo.

    Si no existe el JSON (modelos antiguos), se reconstruye desde el dataset
    como fallback y se registra una advertencia.
    """
    global _label_mapping, _reverse_mapping

    if _label_mapping is not None:
        return _label_mapping, _reverse_mapping

    with _mapping_lock:
        if _label_mapping is not None:
            return _label_mapping, _reverse_mapping

        mapping_path = Path(settings.LABEL_MAPPING_PATH)
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as file:
                raw = json.load(file)
            _label_mapping = {
                str(category): int(index)
                for category, index in raw.items()
            }
            _reverse_mapping = {
                index: category
                for category, index in _label_mapping.items()
            }
            logger.info("Mapeo de etiquetas cargado desde %s", mapping_path)
            return _label_mapping, _reverse_mapping

        logger.warning(
            "No existe %s; reconstruyendo el mapeo desde el dataset.",
            mapping_path,
        )
        from src.datasets.data_loader import DatasetLoader
        from src.datasets.preprocessor import DatasetPreprocessor

        loader = DatasetLoader()
        df = loader.load_training_data()
        preprocessor = DatasetPreprocessor()
        preprocessor.prepare_classification_data(
            df, "Resume Text", "Category"
        )
        _label_mapping, _reverse_mapping = preprocessor.get_label_mapping()

    return _label_mapping, _reverse_mapping


def load_classifier():
    """
    Carga el modelo una sola vez.

    Si no hay modelo válido, lanza ``RuntimeError``: es preferible mostrar un
    error a predecir con pesos aleatorios.
    """
    global _classifier

    if _classifier is not None:
        return _classifier

    with _classifier_lock:
        if _classifier is not None:
            return _classifier

        if not ensure_model_exists():
            raise RuntimeError(
                "El modelo BERT no está disponible. "
                "Revisa la descarga o configura MODEL_PATH."
            )

        from src.models.bert_classifier import BertClassifierModel

        label_map, _ = load_mappings()
        classifier = BertClassifierModel(
            num_classes=len(label_map),
            model_name=os.environ.get("BERT_MODEL", "bert-base-uncased"),
            device="cpu",
        )
        classifier.load_model(str(settings.MODEL_PATH))
        classifier.model_loaded = True
        _classifier = classifier
        logger.info(
            "Modelo cargado: %s clases, dispositivo %s",
            classifier.num_classes,
            classifier.device,
        )

    return _classifier


def get_model_metadata(classifier=None) -> dict:
    """Devuelve la arquitectura sin forzar la carga del modelo."""
    if classifier is not None:
        config = classifier.model.bert.config
        return {
            "total_params": sum(
                parameter.numel() for parameter in classifier.model.parameters()
            ),
            "num_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_attention_heads,
            "vocab_size": config.vocab_size,
            "classifier_classes": classifier.num_classes,
        }

    try:
        from src.utils.config import BERT_CONFIG
    except Exception:
        BERT_CONFIG = {
            "total_params": 110_000_000,
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "vocab_size": 30_522,
        }

    return {
        "total_params": BERT_CONFIG["total_params"],
        "num_layers": BERT_CONFIG["num_layers"],
        "hidden_size": BERT_CONFIG["hidden_size"],
        "num_heads": BERT_CONFIG["num_heads"],
        "vocab_size": BERT_CONFIG["vocab_size"],
    }


# ====================================================================
# DATASET Y MÉTRICAS
# ====================================================================

@lru_cache(maxsize=1)
def get_dataset_info() -> dict:
    """Lee el dataset una sola vez por proceso."""
    try:
        import pandas as pd

        df = pd.read_csv(settings.DATASET_PATH, usecols=["Category"])
        counts = df["Category"].value_counts()
        return {
            "total_samples": int(len(df)),
            "category_counts": {
                str(category): int(count)
                for category, count in counts.items()
            },
        }
    except Exception:
        logger.exception("No se pudo leer el dataset de entrenamiento.")
        return {
            "total_samples": sum(DEFAULT_CATEGORY_COUNTS.values()),
            "category_counts": dict(DEFAULT_CATEGORY_COUNTS),
        }


def get_training_metrics() -> dict:
    """Carga las métricas reales del entrenamiento si existen."""
    path = Path(settings.METRICS_PATH)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as file:
                metrics = json.load(file)
            metrics.setdefault("illustrative", False)
            return metrics
        except Exception:
            logger.exception("No se pudieron leer las métricas del modelo.")
    return dict(DEFAULT_METRICS)


# ====================================================================
# PDF / OCR
# ====================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extrae texto de un PDF.

    Primero intenta con el texto embebido (rápido y exacto). Solo usa OCR en
    páginas que no tienen texto suficiente o en PDFs escaneados.
    """
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    else:
        import platform

        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
        else:
            pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    parts = []
    try:
        with fitz.open(pdf_file) as document:
            total_pages = len(document)
            if total_pages > settings.MAX_PDF_PAGES:
                logger.warning(
                    "PDF con %s páginas; se procesarán solo %s.",
                    total_pages,
                    settings.MAX_PDF_PAGES,
                )

            for page_number, page in enumerate(
                document, start=1
            ):
                if page_number > settings.MAX_PDF_PAGES:
                    break

                page_text = (page.get_text("text") or "").strip()

                if len(page_text) < 50:
                    matrix = fitz.Matrix(2, 2)
                    pixmap = page.get_pixmap(
                        matrix=matrix, colorspace=fitz.csGRAY
                    )
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    try:
                        page_text = pytesseract.image_to_string(
                            image, lang=settings.OCR_LANG
                        ).strip()
                    finally:
                        image.close()

                if page_text:
                    parts.append(page_text)

        text = "\n".join(parts)
        logger.info("Texto extraído: %s caracteres", len(text))
        return text
    except Exception:
        logger.exception("Error extrayendo texto del PDF.")
        return ""


# ====================================================================
# IDIOMA / TRADUCCIÓN
# ====================================================================

def is_english(text: str, threshold: float = 0.5) -> bool:
    """Detecta inglés con langdetect y fallback heurístico."""
    if not text or not text.strip():
        return True

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text[:1000]) == "en"
    except Exception:
        words = re.findall(r"[a-z]+", text.lower())
        if not words:
            return True
        english_words = {
            "the", "and", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "with", "by", "a", "an", "i", "you", "he",
            "she", "it", "we", "they", "my", "your", "his", "her", "its",
            "our", "their", "software", "engineer", "developer",
            "programmer", "systems", "technical", "support", "experience",
            "education", "skills", "python", "java", "javascript",
            "database", "web", "network", "manager", "analyst", "designer",
            "data", "project", "management", "team", "working", "knowledge",
            "professional", "work", "company",
        }
        english_count = sum(1 for word in words if word in english_words)
        return (english_count / len(words)) > threshold


def _split_text(text: str, max_length: int = 4500):
    """Divide el texto en trozos sin cortar palabras."""
    chunks = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(current) + len(sentence) + 1 <= max_length:
            current = f"{current} {sentence}".strip()
            continue

        if current:
            chunks.append(current)
            current = ""

        while len(sentence) > max_length:
            cut = sentence.rfind(" ", 0, max_length)
            cut = cut if cut > 0 else max_length
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        current = sentence

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def translate_to_english(text: str) -> str:
    """Traduce al inglés respetando el idioma de origen."""
    if not text or len(text.strip()) < 20 or is_english(text):
        return text

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target="en")
        chunks = _split_text(text)
        translated = [
            translator.translate(chunk) or chunk
            for chunk in chunks
        ]
        return " ".join(translated)
    except Exception:
        logger.exception("Error traduciendo el texto; se usará el original.")
        return text


# ====================================================================
# PREDICCIÓN
# ====================================================================

def extract_keywords(text: str, top_n: int = 10):
    """Extrae las palabras más frecuentes, excluyendo stopwords."""
    text_lower = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", text_lower)
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "been",
        "were", "being", "with", "this", "that", "from", "they", "will",
        "what", "when", "your", "more", "about", "also", "into", "year",
        "some", "could", "them", "than", "then", "now", "look", "only",
        "come", "its", "over", "such", "take", "just",
    }
    words = [word for word in words if word not in stop_words]
    return Counter(words).most_common(top_n)


def predict_category(text: str, classifier):
    """Ejecuta el pipeline completo de inferencia."""
    import torch

    label_map, reverse_map = load_mappings()

    if not text or len(text.strip()) < 10:
        return None, []
    if not getattr(classifier, "model_loaded", False):
        raise RuntimeError("El modelo no está cargado correctamente.")

    text_en = translate_to_english(text)

    from src.preprocessing.nlp_processor import get_nlp_processor

    nlp = get_nlp_processor()
    nlp_result = nlp.process(
        text_en,
        include_tokens=False,
        include_sentences=False,
        include_noun_chunks=False,
        include_dependencies=False,
    )
    sections = nlp.extract_sections(text_en)

    relevant_sections = ""
    for section_name in ("experience", "education", "skills", "projects"):
        content = sections.get(section_name)
        if content:
            relevant_sections += f"{section_name.upper()}: {content} "

    if not relevant_sections.strip():
        relevant_sections = text_en

    tech_entities = nlp_result["entities"].get("TECH", [])
    skills_from_ner = nlp.get_skills_from_ner(text_en, nlp_result=nlp_result)
    all_skills = sorted(set(tech_entities + skills_from_ner))

    org_entities = nlp_result["entities"].get("ORG", [])[:5]
    keywords_text = extract_keywords(text_en, top_n=15)
    keywords_list = [keyword for keyword, _ in keywords_text]

    if all_skills:
        relevant_sections += "[TECH: " + ", ".join(all_skills) + "] "
    if keywords_list:
        relevant_sections += "[KEYWORDS: " + ", ".join(keywords_list) + "] "
    if org_entities:
        relevant_sections += "[COMPANIES: " + ", ".join(org_entities) + "] "

    cleaned_text = get_text_cleaner().clean(relevant_sections)

    classifier.model.eval()
    encoding = classifier.tokenizer(
        cleaned_text,
        add_special_tokens=True,
        max_length=512,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(classifier.device)
    attention_mask = encoding["attention_mask"].to(classifier.device)

    with torch.no_grad():
        outputs = classifier.model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    predicted_class = int(probabilities.argmax())
    confidence = float(probabilities[predicted_class] * 100)

    results = [
        {
            "category": reverse_map.get(index, f"Category_{index}"),
            "probability": float(probability * 100),
        }
        for index, probability in enumerate(probabilities)
    ]
    results.sort(key=lambda item: item["probability"], reverse=True)

    skills_found = get_skill_extractor().extract_skills(text_en)
    metadata = get_model_metadata(classifier=classifier)
    dataset_info = get_dataset_info()

    result = {
        "category": reverse_map.get(predicted_class),
        "confidence": confidence,
        "nlp_entities": nlp_result["entities"],
        "nlp_lemmas": nlp_result["lemmatized"][:100],
        "nlp_pos_tags": nlp_result["pos_tags"],
        "nlp_sections": sections,
        "skills_from_ner": all_skills,
        "skills": skills_found,
        "text_extracted": text,
        "text_preprocessed": text_en,
        "text_for_model": cleaned_text,
        "keywords": keywords_list,
        "model_info": {
            "total_training_samples": dataset_info["total_samples"],
            "num_categories": len(label_map),
            "model_loaded": True,
            "parameters": {
                "total_parameters": metadata["total_params"],
                "num_hidden_layers": metadata["num_layers"],
                "hidden_size": metadata["hidden_size"],
                "classifier_layers": metadata["classifier_classes"],
            },
        },
        "top_3_predictions": results[:3],
    }
    return result, results[:10]


# ====================================================================
# CARGA DE ARCHIVOS
# ====================================================================

def _process_upload(cv_file):
    """Valida, guarda, analiza y elimina un PDF subido."""
    if not cv_file.name.lower().endswith(".pdf"):
        raise ValueError("Sube un archivo PDF.")

    if cv_file.size and cv_file.size > settings.MAX_PDF_SIZE:
        max_mb = settings.MAX_PDF_SIZE / (1024 * 1024)
        raise ValueError(f"El PDF supera el tamaño máximo de {max_mb:.0f} MB.")

    header = cv_file.read(5)
    cv_file.seek(0)
    if header != b"%PDF-":
        raise ValueError("El archivo no parece un PDF válido.")

    storage = FileSystemStorage(location=settings.MEDIA_ROOT)
    filename = storage.save(os.path.basename(cv_file.name), cv_file)
    file_path = Path(settings.MEDIA_ROOT) / filename

    try:
        text = extract_text_from_pdf(str(file_path))
        if not text or len(text.strip()) < 10:
            raise ValueError("No se pudo extraer texto del PDF.")

        classifier = load_classifier()
        result, top_categories = predict_category(text, classifier)
        if result is None:
            raise ValueError("El texto del PDF es demasiado corto para analizarlo.")
        return result, top_categories
    finally:
        file_path.unlink(missing_ok=True)


# ====================================================================
# VISTAS
# ====================================================================

def home(request):
    result = request.session.pop("cv_result", None)
    top_categories = request.session.pop("cv_top_categories", None)
    error = request.session.pop("cv_error", None)

    response = render(
        request,
        "analyzer/index.html",
        {
            "result": result,
            "top_categories": top_categories,
            "error": error,
        },
    )
    return _no_cache(response)


def upload_cv(request):
    if request.method == "GET":
        return _no_cache(redirect("home"))

    cv_file = request.FILES.get("cv_file")
    if not cv_file:
        request.session["cv_error"] = "Selecciona un archivo PDF."
        return _no_cache(redirect("home"))

    try:
        result, top_categories = _process_upload(cv_file)
        request.session["cv_result"] = result
        request.session["cv_top_categories"] = top_categories
    except ValueError as error:
        request.session["cv_error"] = str(error)
    except Exception:
        logger.exception("Error procesando el CV.")
        request.session["cv_error"] = (
            "No se pudo analizar el CV. Inténtalo de nuevo."
        )

    return _no_cache(redirect("home"))


def analyze_pdf(request):
    result = None
    top_categories = []
    error = None

    if request.method == "POST":
        cv_file = request.FILES.get("cv_file")
        if not cv_file:
            error = "Selecciona un archivo PDF."
        else:
            try:
                result, top_categories = _process_upload(cv_file)
            except ValueError as exc:
                error = str(exc)
            except Exception:
                logger.exception("Error procesando el CV.")
                error = "No se pudo analizar el CV. Inténtalo de nuevo."

    response = render(
        request,
        "analyzer/analyze.html",
        {
            "result": result,
            "top_categories": top_categories,
            "error": error,
        },
    )
    return _no_cache(response)


def model_info(request):
    dataset_info = get_dataset_info()
    category_counts = dataset_info["category_counts"]

    ordered_counts = sorted(
        category_counts.items(), key=lambda item: item[1], reverse=True
    )[:15]
    category_labels = [category for category, _ in ordered_counts]
    category_values = [count for _, count in ordered_counts]

    label_map, _ = load_mappings()
    metadata = get_model_metadata()
    total_samples = dataset_info["total_samples"]

    return render(
        request,
        "analyzer/model_info.html",
        {
            "total_samples": total_samples,
            "train_samples": int(total_samples * 0.8),
            "val_samples": int(total_samples * 0.2),
            "num_categories": len(label_map),
            "categories": list(label_map.keys()),
            "category_labels": category_labels,
            "category_values": category_values,
            "category_distribution": category_counts,
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": "2e-5",
            "max_length": 512,
            "model_params": {
                "total_params": metadata["total_params"],
                "num_layers": metadata["num_layers"],
                "hidden_size": metadata["hidden_size"],
                "num_heads": metadata["num_heads"],
                "vocab_size": metadata["vocab_size"],
            },
            "classifier_classes": len(label_map),
            "metrics": get_training_metrics(),
        },
    )


def health(request):
    return JsonResponse(
        {
            "status": "ok",
            "model_exists": Path(settings.MODEL_PATH).exists(),
            "model_loaded": _classifier is not None,
        }
    )
