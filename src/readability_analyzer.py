"""
E-Reader Text Analytics Engine - Readability Analyzer Module

Calculates readability metrics including Flesch Reading Ease, Flesch-Kincaid
Grade Level, vocabulary richness, and word-level statistics.
"""

import re
from typing import Dict, Optional
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False


class ReadabilityAnalyzer:
    """Calculates readability metrics for text."""

    def analyze(self, text: str) -> Dict[str, object]:
        """Calculate readability metrics for the given text.

        Args:
            text: Original text with punctuation preserved.

        Returns:
            Dictionary of readability metrics.
        """
        if not text or not text.strip():
            return self._empty_result("No text available for readability analysis.")

        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        # Filter to actual words (not punctuation-only tokens)
        words = [w for w in words if re.search(r'[a-zA-Z0-9]', w)]

        word_count = len(words)
        sentence_count = len(sentences)

        if word_count == 0:
            return self._empty_result("No words found in text.")

        if sentence_count == 0:
            return self._empty_result("No sentences detected in text.")

        # Basic metrics
        unique_words = set(w.lower() for w in words)
        unique_count = len(unique_words)
        avg_word_length = sum(len(w) for w in words) / word_count
        avg_words_per_sentence = word_count / sentence_count
        vocab_richness = unique_count / word_count

        result = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'unique_word_count': unique_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'vocabulary_richness': round(vocab_richness, 4),
            'flesch_reading_ease': None,
            'flesch_kincaid_grade': None,
            'error': None,
        }

        # Textstat-based metrics
        if TEXTSTAT_AVAILABLE:
            try:
                fre = textstat.flesch_reading_ease(text)
                fkg = textstat.flesch_kincaid_grade(text)
                result['flesch_reading_ease'] = round(fre, 2)
                result['flesch_kincaid_grade'] = round(fkg, 2)
            except Exception as e:
                result['error'] = f"Readability formulas could not be applied: {e}"
        else:
            result['error'] = ("textstat library not available. "
                               "Install it with: pip install textstat")

        return result

    def get_reading_level_description(self, flesch_score: Optional[float]) -> str:
        """Return a human-readable description of the Flesch Reading Ease score.

        Args:
            flesch_score: The Flesch Reading Ease score.

        Returns:
            Description string.
        """
        if flesch_score is None:
            return "Score not available"

        if flesch_score >= 90:
            return "Very Easy (5th grade)"
        elif flesch_score >= 80:
            return "Easy (6th grade)"
        elif flesch_score >= 70:
            return "Fairly Easy (7th grade)"
        elif flesch_score >= 60:
            return "Standard (8th-9th grade)"
        elif flesch_score >= 50:
            return "Fairly Difficult (10th-12th grade)"
        elif flesch_score >= 30:
            return "Difficult (College level)"
        else:
            return "Very Confusing (College graduate)"

    def _empty_result(self, message: str) -> Dict[str, object]:
        """Return an empty metrics dictionary with an error message."""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'unique_word_count': 0,
            'avg_word_length': 0.0,
            'avg_words_per_sentence': 0.0,
            'vocabulary_richness': 0.0,
            'flesch_reading_ease': None,
            'flesch_kincaid_grade': None,
            'error': message,
        }
