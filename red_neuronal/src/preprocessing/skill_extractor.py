"""
=================================================================
EXTRACTOR DE HABILIDADES (SKILLS)
Universidad - Detección de Skills en Currículums

Este módulo detecta diferentes tipos de habilidades en textos
de currículums:

Categorías de skills detectados:
1. Technical (programming languages, data science, AI/ML)
2. Frameworks (Django, React, Spring, etc.)
3. Databases (SQL, MongoDB, PostgreSQL, etc.)
4. Cloud (AWS, Azure, Docker, Kubernetes)
5. Soft skills (leadership, communication, etc.)
6. Languages (english, spanish, etc.)
7. Tools (Git, Jenkins, etc.)

Método:
- Búsqueda por palabras clave (keyword matching)
- Comparación con lista de skills opcional

=================================================================
"""

import re                      # Expresiones regulares
from typing import List, Dict, Set  # Type hints


def contains_skill(text: str, term: str, case_sensitive: bool = False) -> bool:
    """
    Comprueba si ``term`` aparece como palabra/frase completa.

    Evita falsos positivos como ``go`` dentro de ``good``, ``ai`` dentro de
    ``email`` o ``java`` dentro de ``javascript``.
    """
    if not term:
        return False
    haystack = text if case_sensitive else text.lower()
    needle = term if case_sensitive else term.lower()
    pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return re.search(pattern, haystack) is not None


class SkillExtractor:
    """
    Extrator de habilidades desde texto de currículum.
    
    Detecta skills usando búsqueda de palabras clave en el texto.
    
    Atributos:
        skills_list: lista adicional de skills (opcional)
        tech_keywords: diccionario de categorías técnicas
        soft_skills: lista de habilidades blandas
        languages: lista de idiomas
    
    Ejemplo de uso:
        extractor = SkillExtractor()
        skills = extractor.extract_skills("I know Python and JavaScript...")
        # {'technical': ['python', 'javascript'], 'soft': [], ...}
    """
    
    def __init__(self, skills_list: List[str] = None):
        """
        Inicializa el extractor de skills.
        
        Args:
            skills_list: lista adicional de skills a buscar (opcional)
                         Puede venir del dataset skills_list.csv
        """
        # Lista adicional de skills desde dataset
        self.skills_list = skills_list or []
        
        # Construir patrones de búsqueda
        self._build_skill_patterns()

    def _build_skill_patterns(self):
        """
        Define las categorías y palabras clave de skills.
        
        Estructura:
        - tech_keywords: dict con categorías técnicas
          - programming: lenguajes de programación
          - frontend: frameworks de frontend
          - backend: frameworks de backend
          - database: sistemas de bases de datos
          - cloud: servicios cloud y DevOps
          - data_science: librerías de datos
          - devops: herramientas de DevOps
          - ai_ml: técnicas de IA/ML
        
        - soft_skills: habilidades blandas
        - languages: idiomas
        """
        # Palabras clave técnicas organizadas por categoría
        self.tech_keywords = {
            # Lenguajes de programación
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 
                          'ruby', 'go', 'rust', 'typescript', 'php', 'swift', 'kotlin'],
            
            # Frameworks de frontend
            'frontend': ['html', 'css', 'react', 'angular', 'vue', 
                        'bootstrap', 'tailwind', 'sass', 'jquery'],
            
            # Frameworks de backend
            'backend': ['nodejs', 'django', 'flask', 'spring', 'rails', 
                       'express', 'laravel', 'asp.net'],
            
            # Bases de datos
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 
                        'redis', 'elasticsearch', 'cassandra'],
            
            # Cloud y DevOps
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 
                     'terraform', 'cloudformation'],
            
            # Data Science y ML
            'data_science': ['tensorflow', 'pytorch', 'pandas', 'numpy', 
                            'scikit-learn', 'keras', 'spark', 'tableau'],
            
            # Herramientas DevOps
            'devops': ['jenkins', 'gitlab', 'github', 'ansible', 'puppet', 
                      'circleci', 'travis', 'nginx'],
            
            # AI y Machine Learning
            'ai_ml': ['machine learning', 'deep learning', 'nlp', 
                     'computer vision', 'neural networks', 'ai'],
        }
        
        # Habilidades blandas (soft skills)
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 
            'analytical', 'creativity', 'adaptability', 'time management', 
            'organization', 'critical thinking', 'decision making', 
            'conflict resolution', 'project management', 'agile', 'scrum', 
            'mentoring', 'presentation'
        ]
        
        # Idiomas
        self.languages = [
            'english', 'spanish', 'french', 'german', 'chinese', 'japanese',
            'korean', 'portuguese', 'italian', 'russian', 'arabic', 'hindi'
        ]

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae todos los skills del texto.
        
        Proceso:
        1. Convertir texto a minúsculas
        2. Buscar cada palabra clave en el texto
        3. Clasificar en categorías correspondientes
        4. Eliminar duplicados
        
        Args:
            texto del currículum
        
        Returns:
            dict: {
                'technical': [...],
                'soft': [...],
                'languages': [...],
                'frameworks': [...],
                'tools': [...],
                'databases': [...],
                'cloud': [...]
            }
        
        Ejemplo:
            Input: "Python developer with React and AWS experience"
            Output: {
                'technical': ['python'],
                'frameworks': ['react'],
                'cloud': ['aws'],
                ...
            }
        """
        text_lower = text.lower()
        
        # Inicializar diccionario de salida
        skills = {
            'technical': [],      # Skills técnicos generales
            'soft': [],          # Habilidades blandas
            'languages': [],     # Idiomas
            'frameworks': [],    # Frameworks web
            'tools': [],         # Herramientas
            'databases': [],     # Bases de datos
            'cloud': []          # Servicios cloud
        }
        
        found_skills = set()  # Para evitar duplicados

        # ========== BUSCAR SKILLS TÉCNICOS ==========
        # Iterar sobre cada categoría de tech_keywords
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                # Si la palabra está en el texto como término completo
                if contains_skill(text_lower, keyword):
                    # Clasificar según la categoría
                    if category in ['frontend', 'backend']:
                        if keyword not in found_skills:
                            skills['frameworks'].append(keyword)
                            found_skills.add(keyword)
                    elif category == 'database':
                        if keyword not in found_skills:
                            skills['databases'].append(keyword)
                            found_skills.add(keyword)
                    elif category == 'cloud':
                        if keyword not in found_skills:
                            skills['cloud'].append(keyword)
                            found_skills.add(keyword)
                    elif category == 'data_science' or category == 'ai_ml':
                        if keyword not in found_skills:
                            skills['technical'].append(keyword)
                            found_skills.add(keyword)
                    else:  # programming
                        if keyword not in found_skills:
                            skills['technical'].append(keyword)
                            found_skills.add(keyword)

        # ========== BUSCAR DESDE LISTA EXTERNA ==========
        # Si tenemos lista adicional de skills desde dataset
        if self.skills_list:
            for skill in self.skills_list:
                skill_lower = skill.lower()
                if (
                    contains_skill(text_lower, skill_lower)
                    and skill_lower not in found_skills
                ):
                    skills['technical'].append(skill)
                    found_skills.add(skill_lower)

        # ========== BUSCAR SOFT SKILLS ==========
        for soft_skill in self.soft_skills:
            if contains_skill(text_lower, soft_skill):
                skills['soft'].append(soft_skill)

        # ========== BUSCAR IDIOMAS ==========
        for language in self.languages:
            if contains_skill(text_lower, language):
                skills['languages'].append(language)

        # Eliminar duplicados en cada categoría
        return {k: list(set(v)) for k, v in skills.items()}

    def extract_from_resume(self, text: str) -> List[str]:
        """
        Extrae skills y retorna lista plana (sin categorías).
        
        Args:
            texto del currículum
        
        Returns:
            list: todos los skills encontrados (sin duplicados)
        
        Útil para:
        - Mostrar skills en la UI
        - Comparar con otros currículums
        """
        all_skills = []
        extracted = self.extract_skills(text)
        
        # Unir todas las categorías en una lista
        for category_skills in extracted.values():
            all_skills.extend(category_skills)
        
        # Eliminar duplicados
        return list(set(all_skills))