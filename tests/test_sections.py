from src.preprocessing.section_extractor import extract_sections


def test_extracts_sections_between_headings():
    text = """John Doe
Software Engineer

EXPERIENCE
- Built Python services with Django.
- Led a team of 4 engineers.

EDUCATION
BS Computer Science, University of Technology

SKILLS
Python, Django, PostgreSQL, AWS

PROJECTS
CV Analyzer
"""
    sections = extract_sections(text)

    assert "Built Python services" in sections["experience"]
    assert "University of Technology" in sections["education"]
    assert "PostgreSQL" in sections["skills"]
    assert "CV Analyzer" in sections["projects"]
    assert "University of Technology" not in sections["experience"]


def test_does_not_overlap_sections():
    text = """EXPERIENCE
Experience with Python.
EDUCATION
Education in Computer Science.
SKILLS
Skills: Python, SQL.
"""
    sections = extract_sections(text)

    assert "Education" not in sections["experience"]
    assert "Skills" not in sections["education"]


def test_returns_empty_sections_without_headings():
    sections = extract_sections("Just a plain CV without headings.")

    assert all(value == "" for value in sections.values())
