"""
E-Reader Text Analytics Engine - Text Processor Module

Handles loading, cleaning, tokenizing, and preprocessing text from book files.
Preserves original text for readability/sentiment while providing cleaned tokens
for topic modeling.
"""

import re
import os
import nltk
from typing import Dict, List, Optional, Tuple

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data():
    """Download required NLTK resources if not already present."""
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# Initialize NLTK data on module load
ensure_nltk_data()


class TextProcessor:
    """Processes raw text files for NLP analysis.

    Provides both cleaned/tokenized text for topic modeling and
    preserved original text for readability and sentiment analysis.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def load_text(self, file_path: str) -> str:
        """Read a TXT file with encoding fallback.

        Tries UTF-8 first, then falls back to latin-1 which can read
        any byte sequence without errors.

        Args:
            file_path: Path to the text file.

        Returns:
            The raw text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or unreadable.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.lower().endswith('.txt'):
            raise ValueError(f"Unsupported file type. Please select a .txt file.")

        # Try UTF-8 first, then latin-1 as fallback
        for encoding in ('utf-8', 'latin-1'):
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                if text.strip():
                    return text
                else:
                    raise ValueError("The file is empty or contains only whitespace.")
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Unable to read file with supported encodings: {file_path}")

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure.

        Collapses runs of spaces/tabs to single spaces within lines,
        preserves newlines for chapter detection.
        """
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Collapse multiple spaces to one
        text = re.sub(r'[ ]+', ' ', text)
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Collapse more than 2 consecutive newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def get_basic_stats(self, text: str) -> Dict[str, int]:
        """Calculate basic text statistics from original text.

        Args:
            text: The original (non-cleaned) text.

        Returns:
            Dictionary with character_count, word_count, sentence_count.
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        # Filter to actual words (not pure punctuation)
        words = [w for w in words if re.search(r'[a-zA-Z0-9]', w)]

        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
        }

    def clean_for_topic_modeling(self, text: str) -> str:
        """Clean text specifically for topic modeling.

        Lowercases, removes non-alphabetic characters, and normalizes whitespace.
        This is a destructive transformation only for topic modeling tokens.
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common English stop words from token list."""
        return [w for w in tokens if w.lower() not in self.stop_words and len(w) > 1]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base forms."""
        return [self.lemmatizer.lemmatize(w) for w in tokens]

    def get_tokens_for_topic_modeling(self, text: str) -> List[str]:
        """Get fully preprocessed tokens suitable for LDA topic modeling.

        Pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
        """
        cleaned = self.clean_for_topic_modeling(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize(filtered)
        return lemmatized

    def detect_title(self, file_path: str, text: str) -> str:
        """Attempt to detect the book title.

        Uses the filename as the primary source, stripping extension.
        """
        basename = os.path.splitext(os.path.basename(file_path))[0]
        # Clean up common filename patterns
        title = basename.replace('_', ' ').replace('-', ' ')
        # Title-case it
        title = title.title()
        return title

    def process(self, file_path: str) -> Dict:
        """Full processing pipeline for a book file.

        Returns a dictionary containing:
            - raw_text: Original text preserved for readability/sentiment
            - normalized_text: Whitespace-normalized text
            - tokens_for_lda: Cleaned tokens for topic modeling
            - basic_stats: Character, word, sentence counts
            - file_path: Source file path
            - title: Detected book title
        """
        raw_text = self.load_text(file_path)
        normalized_text = self.normalize_whitespace(raw_text)
        basic_stats = self.get_basic_stats(normalized_text)
        tokens_for_lda = self.get_tokens_for_topic_modeling(normalized_text)
        title = self.detect_title(file_path, normalized_text)

        return {
            'raw_text': raw_text,
            'normalized_text': normalized_text,
            'tokens_for_lda': tokens_for_lda,
            'basic_stats': basic_stats,
            'file_path': file_path,
            'title': title,
        }