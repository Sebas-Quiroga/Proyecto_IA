"""
=================================================================
PROCESADOR NLP AVANZADO (NER, LEMMATIZACIÓN, EXTRACCIÓN DE SECCIONES)
Universidad - Pipeline de Procesamiento de Lenguaje Natural

Este módulo implementa un pipeline completo de NLP usando spaCy
para análisis de currículums.

Funcionalidades:
1. NER (Named Entity Recognition): Detectar personas, organizaciones, fechas
2. Detección de tecnologías: Python, Java, AWS, Docker, etc.
3. Extracción de secciones: Experience, Education, Skills, Projects
4. Lemmatización: Reducir palabras a su forma raíz
5. POS tagging: Etiquetado de categorías gramaticales
6. Dependency parsing: Análisis sintáctico

Dependencias:
- spacy: Framework de NLP industrial
- re: Expresiones regulares para detección de tecnologías

=================================================================
"""

import re                      # Expresiones regulares para detección de tech
import threading              # Inicialización segura del singleton
from typing import Dict, List, Tuple, Optional  # Type hints

import spacy                  # Framework de NLP (spaCy)

from .section_extractor import extract_sections as _extract_sections
from .skill_extractor import contains_skill


# Tecnologías detectables. Se comparan como término completo para evitar
# falsos positivos (por ejemplo, "Go" dentro de "Google").
TECH_TERMS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "Go",
    "Rust", "Kotlin", "Swift", "PHP", "R", "Scala", "Perl", "Django", "Flask",
    "Spring", "Rails", "Express", "NestJS", "React", "Vue", "Angular", "Node.js",
    "Node", "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle",
    "Elasticsearch", "Cassandra", "AWS", "GCP", "Azure", "Docker", "Kubernetes",
    "Terraform", "Git", "GitHub", "GitLab", "Jira", "TensorFlow", "PyTorch",
    "Keras", "Scikit-learn", "Pandas", "NumPy", "HTML", "CSS", "Bootstrap",
    "Tailwind", "Linux", "Jenkins", "Ansible", "Spark", "Hadoop", "Tableau",
    "Power BI", "Excel", "Machine Learning", "Deep Learning", "NLP",
]

# Alias que deben mostrarse con una forma canónica.
TECH_CANONICAL = {
    "scikit": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
    "nodejs": "Node.js",
    "node": "Node.js",
    "powerbi": "Power BI",
    "power bi": "Power BI",
}


class NLPProcessor:
    """
    Procesador completo de NLP para currículums.
    
    Utiliza spaCy con el modelo en_core_web_sm para:
    - Reconocimiento de entidades nombradas (NER)
    - Lemmatización
    - POS tagging
    - Análisis de dependencias
    
    Adicionalmente detecta:
    - Tecnologías (Python, Java, AWS, etc.) via regex
    - Secciones del CV (experience, education, skills)
    - Skills técnicos
    
    Args:
        model: nombre del modelo de spaCy (default: en_core_web_sm)
    
    Attributes:
        nlp: instancia del pipeline de spaCy
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Inicializa el procesador de NLP.
        
        Proceso:
        1. Intenta cargar el modelo de spaCy
        2. Si no existe, lo descarga automáticamente
        3. Configura el límite de longitud máxima
        
        Dependencias:
            - spacy.load: cargar modelo pre-entrenado
            - subprocess: para descargar modelo si no existe
        
        Modelo utilizado:
        - en_core_web_sm: modelo pequeño de inglés (12 MB)
          - NER: PERSON, ORG, GPE, DATE, LOC, etc.
          - POS tags: NOUN, VERB, ADJ, etc.
          - Lemmatización: inglés
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Si el modelo no está instalado, descargarlo
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)
        
        # Aumentar límite de longitud para textos largos
        self.nlp.max_length = 2000000
    
    def process(
        self,
        text: str,
        include_tokens: bool = True,
        include_sentences: bool = True,
        include_noun_chunks: bool = True,
        include_dependencies: bool = True,
    ) -> Dict:
        """
        Procesa un texto y retorna información NLP completa.
        
        Pipeline:
        1. Truncar a 100k caracteres (límite de spaCy)
        2. Procesar con el pipeline de spaCy
        3. Extraer: entidades, lemmas, tokens, oraciones, POS, noun chunks, dependencias
        
        Args:
            text: texto del currículum
        
        Returns:
            dict: {
                'entities': {PERSON: [], ORG: [], TECH: [], ...},
                'lemmatized': [palabras lematizadas],
                'tokens': [{text, lemma, pos, tag}, ...],
                'sentences': [oraciones],
                'pos_tags': {NOUN: count, VERB: count, ...},
                'noun_chunks': [grupos nominales],
                'dependencies': [{text, dep, head}, ...]
            }
        
        Dependencias internas:
            - _extract_entities: NER
            - _get_lemmas: lematización
            - _get_tokens: tokens con metadatos
            - _get_sentences: segmentación
            - _get_pos_tags: conteo de POS
            - _get_noun_chunks: grupos nominales
            - _get_dependencies: análisis sintáctico
        """
        # Limitar longitud del texto (spaCy tiene límite de ~1M caracteres)
        doc = self.nlp(text[:100000])

        result = {
            'entities': self._extract_entities(doc),
            'lemmatized': self._get_lemmas(doc),
            'pos_tags': self._get_pos_tags(doc),
        }
        if include_tokens:
            result['tokens'] = self._get_tokens(doc)
        if include_sentences:
            result['sentences'] = self._get_sentences(doc)
        if include_noun_chunks:
            result['noun_chunks'] = self._get_noun_chunks(doc)
        if include_dependencies:
            result['dependencies'] = self._get_dependencies(doc)
        return result
    
    def _extract_entities(self, doc) -> Dict:
        """
        Extrae entidades nombradas y detecta tecnologías.
        
        Tipos de entidades detectadas:
        - PERSON: nombres de personas
        - ORG: organizaciones (empresas)
        - GPE: países/ciudades
        - DATE: fechas
        - TECH: tecnologías (detección personalizada con regex)
        - EDUCATION: secciones educativas (detección por palabras clave)
        
        Proceso:
        1. Iterar sobre entidades de spaCy (doc.ents)
        2. Detectar tecnologías con patrones regex
        3. Detectar educación con palabras clave
        
        Args:
            doc: documento procesado por spaCy
        
        Returns:
            dict: entidades por tipo
        
        Dependencias:
            - re.findall: búsqueda de patrones
            - doc.ents: iterator de entidades spaCy
            - ent.label_: tipo de entidad
            - ent.text: texto de la entidad
        """
        # Inicializar diccionario de entidades
        entities = {'PERSON': [], 'ORG': [], 'GPE': [], 'DATE': [], 
                   'SKILL': [], 'TECH': [], 'EDUCATION': [], 'LOCATION': []}
        
        # Extraer entidades del modelo spaCy
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # =================================================================
        # DETECCIÓN DE TECNOLOGÍAS
        # =================================================================
        # SpaCy no tiene entidad TECH, así que comparamos una lista de
        # tecnologías conocidas como términos completos.
        text_lower = doc.text.lower()
        found_tech = set()

        # "Go" y "R" son ambiguos en minúsculas ("go to", "r"). Exigir la
        # grafía exacta reduce falsos positivos.
        ambiguous = {"Go", "R"}
        for term in TECH_TERMS:
            haystack = doc.text if term in ambiguous else text_lower
            if contains_skill(haystack, term, case_sensitive=term in ambiguous):
                canonical = TECH_CANONICAL.get(term.lower(), term)
                found_tech.add(canonical)

        entities['TECH'] = sorted(found_tech)
        
        # =================================================================
        # DETECCIÓN DE EDUCACIÓN
        # =================================================================
        # Buscar oraciones que contengan palabras relacionadas con educación
        edu_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 
                        'phd', 'certificate', 'diploma', 'education']
        
        education_entities = []
        for sent in doc.sents:
            # Si la oración contiene palabras clave educativas
            if any(kw in sent.text.lower() for kw in edu_keywords):
                education_entities.append(sent.text.strip())
        
        entities['EDUCATION'] = education_entities[:5]  # Máximo 5
        
        # Limitar a 20 entidades por tipo
        return {k: list(set(v))[:20] for k, v in entities.items()}
    
    def _get_lemmas(self, doc) -> List[str]:
        """
        Extrae lemas (forma raíz de las palabras).
        
        Proceso:
        1. Iterar sobre todos los tokens
        2. Excluir stopwords y puntuación
        3. Retornar lemma de cada palabra
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: palabras en su forma raíz
        
        Ejemplo:
            "running" -> "run"
            "developed" -> "develop"
            "technologies" -> "technology"
        
        Dependencias:
            - token.lemma_: forma lematizada
            - token.is_stop: es stopword?
            - token.is_punct: es puntuación?
        """
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    def _get_tokens(self, doc) -> List[Dict]:
        """
        Extrae tokens con metadatos.
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: [{text, lemma, pos, tag}, ...]
        
        Campos:
            - text: texto original
            - lemma: forma raíz
            - pos: categoría gramatical (NOUN, VERB, etc.)
            - tag: etiqueta detallada (NN, VB, etc.)
        """
        return [{'text': token.text, 'lemma': token.lemma_, 
                 'pos': token.pos_, 'tag': token.tag_} for token in doc]
    
    def _get_sentences(self, doc) -> List[str]:
        """
        Segmenta el texto en oraciones.
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: oraciones del texto
        """
        return [sent.text.strip() for sent in doc.sents]
    
    def _get_pos_tags(self, doc) -> Dict:
        """
        Cuenta las categorías gramaticales (POS tags).
        
       POS Tags principales:
        - NOUN: sustantivos
        - VERB: verbos
        - ADJ: adjetivos
        - ADV: adverbios
        - PROPN: nombres propios
        - NUM: números
        
        Args:
            doc: documento spaCy
        
        Returns:
            dict: {POS: cantidad}
        """
        pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PROPN': 0, 'NUM': 0}
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
        return pos_counts
    
    def _get_noun_chunks(self, doc) -> List[str]:
        """
        Extrae grupos nominales (noun chunks).
        
        Un noun chunk es una frase nominal: "the modern software engineer"
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: grupos nominales
        """
        return [chunk.text for chunk in doc.noun_chunks][:30]
    
    def _get_dependencies(self, doc) -> List[Dict]:
        """
        Extrae análisis de dependencias sintácticas.
        
        Muestra la relación entre palabras:
        - nsubj: sujeto nominal
        - dobj: objeto directo
        - prep: preposición
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: [{text, dep, head}, ...]
        """
        return [{'text': token.text, 'dep': token.dep_, 
                 'head': token.head.text} for token in doc[:50]]
    
    # =================================================================
    # EXTRACCIÓN DE SECCIONES DEL CV
    # =================================================================
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extrae secciones relevantes del currículum.

        La implementación vive en ``section_extractor`` para poder probarla
        sin cargar el modelo de spaCy.
        """
        return _extract_sections(text)
    
    # =================================================================
    # EXTRACCIÓN DE SKILLS
    # =================================================================
    
    def get_skills_from_ner(
        self, text: str, nlp_result: Optional[Dict] = None
    ) -> List[str]:
        """
        Extrae habilidades técnicas del texto.

        Si ya se procesó el texto con ``process()``, se puede pasar
        ``nlp_result`` para evitar ejecutar spaCy dos veces.
        """
        result = nlp_result if nlp_result is not None else self.process(text)
        skills = set(result['entities'].get('TECH', []))

        text_lower = text.lower()
        ambiguous = {"Go", "R"}
        for term in TECH_TERMS:
            haystack = text if term in ambiguous else text_lower
            if contains_skill(haystack, term, case_sensitive=term in ambiguous):
                skills.add(TECH_CANONICAL.get(term.lower(), term))

        return sorted(skills)


# ====================================================================
# INSTANCIA GLOBAL (Singleton)
# ====================================================================

# Instancia global del procesador NLP
# Se inicializa una sola vez para evitar recargar el modelo
nlp_processor = None
_nlp_lock = threading.Lock()


def get_nlp_processor() -> NLPProcessor:
    """
    Retorna la instancia global del procesador NLP.

    Usa un lock para evitar que dos requests simultáneos carguen spaCy dos
    veces en el primer acceso.
    """
    global nlp_processor
    if nlp_processor is None:
        with _nlp_lock:
            if nlp_processor is None:
                nlp_processor = NLPProcessor()
    return nlp_processor