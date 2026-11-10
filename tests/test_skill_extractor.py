from src.preprocessing.skill_extractor import SkillExtractor, contains_skill


def test_contains_skill_uses_word_boundaries():
    assert contains_skill("I use Python and Go.", "python")
    assert contains_skill("I use Python and Go.", "Go", case_sensitive=True)
    assert not contains_skill("I am good at teamwork.", "go")
    assert not contains_skill("Send me an email.", "ai")
    assert not contains_skill("I know JavaScript.", "java")


def test_extractor_does_not_report_false_positives():
    extractor = SkillExtractor()
    skills = extractor.extract_skills(
        "I am good at communication and email. I know JavaScript and Python."
    )

    technical = skills["technical"]
    assert "python" in technical
    assert "javascript" in technical
    assert "go" not in technical
    assert "ai" not in technical
    assert "java" not in technical


def test_extractor_detects_technical_and_soft_skills():
    extractor = SkillExtractor()
    skills = extractor.extract_skills(
        "Python developer with React, AWS and leadership experience."
    )

    assert "python" in skills["technical"]
    assert "react" in skills["frameworks"]
    assert "aws" in skills["cloud"]
    assert "leadership" in skills["soft"]
