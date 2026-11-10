"""
Extracción de secciones de un currículum.

Se separa de ``nlp_processor`` para poder probarla sin cargar spaCy.
"""

import re
from typing import Dict, List, Tuple


SECTION_ALIASES = {
    "experience": [
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "experience",
        "employment",
        "career",
    ],
    "education": [
        "education",
        "academic background",
        "academic",
        "studies",
        "university",
        "college",
    ],
    "skills": [
        "technical skills",
        "skills",
        "technologies",
        "competencies",
        "expertise",
        "proficiencies",
    ],
    "projects": [
        "projects",
        "portfolio",
        "achievements",
    ],
    "contact": [
        "contact",
        "contact information",
        "email",
        "phone",
        "address",
        "location",
    ],
}

_ALIAS_TO_SECTION = {
    alias: section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

# Ordenar por longitud para que "work experience" gane sobre "experience".
_ALIASES_SORTED = sorted(_ALIAS_TO_SECTION, key=len, reverse=True)


def _build_heading_regex() -> re.Pattern:
    alternatives = "|".join(re.escape(alias) for alias in _ALIASES_SORTED)
    # Cabecera en una línea propia, con dos puntos opcionales.
    return re.compile(
        rf"^[ \t]*(?P<alias>{alternatives})[ \t]*:?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )


_HEADING_RE = _build_heading_regex()


def _collect_matches(text: str, regex: re.Pattern) -> List[Tuple[int, int, str]]:
    matches: List[Tuple[int, int, str]] = []
    for match in regex.finditer(text):
        alias = match.group("alias").lower().strip()
        section = _ALIAS_TO_SECTION.get(alias)
        if section:
            matches.append((match.start(), match.end(), section))
    matches.sort(key=lambda item: item[0])
    return matches


def extract_sections(text: str) -> Dict[str, str]:
    """
    Divide el texto en secciones usando las cabeceras detectadas.

    Estrategia:
    1. Buscar cabeceras que ocupen una línea completa.
    2. Si no hay ninguna, buscar la primera aparición de cada alias en el texto.
    3. Cortar el contenido entre cabeceras consecutivas, no entre apariciones
       de la misma cabecera.
    """
    if not text or not text.strip():
        return {section: "" for section in SECTION_ALIASES}

    matches = _collect_matches(text, _HEADING_RE)

    if not matches:
        # Fallback: aliases en cualquier parte del texto, con límites de palabra.
        alternatives = "|".join(re.escape(alias) for alias in _ALIASES_SORTED)
        inline_re = re.compile(
            rf"\b(?P<alias>{alternatives})\b", re.IGNORECASE
        )
        matches = _collect_matches(text, inline_re)

    sections = {section: "" for section in SECTION_ALIASES}
    seen = set()

    for index, (start, end, section) in enumerate(matches):
        if section in seen:
            continue
        seen.add(section)

        next_start = len(text)
        for next_match in matches[index + 1 :]:
            if next_match[0] > end:
                next_start = next_match[0]
                break

        sections[section] = text[end:next_start].strip()

    return sections
