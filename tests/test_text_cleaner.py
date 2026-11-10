from src.preprocessing.text_cleaner import TextCleaner


def test_clean_normalizes_text():
    cleaner = TextCleaner()
    result = cleaner.clean("  I'm a Software Engineer! 😊  ")
    assert result == "i am a software engineer"


def test_clean_returns_empty_for_empty_input():
    cleaner = TextCleaner()
    assert cleaner.clean("   ") == ""
    assert cleaner.clean(None) == ""
