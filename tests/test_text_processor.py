"""Tests for the TextProcessor module."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    """Test suite for TextProcessor."""

    def setUp(self):
        self.processor = TextProcessor()

    def test_load_valid_file(self):
        """Test loading a valid UTF-8 text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                          delete=False, encoding='utf-8') as f:
            f.write("Hello world. This is a test.")
            f.flush()
            text = self.processor.load_text(f.name)
            self.assertIn("Hello world", text)
        os.unlink(f.name)

    def test_load_empty_file(self):
        """Test that empty files raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                          delete=False, encoding='utf-8') as f:
            f.write("")
            f.flush()
            with self.assertRaises(ValueError):
                self.processor.load_text(f.name)
        os.unlink(f.name)

    def test_load_nonexistent_file(self):
        """Test that missing files raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_text("nonexistent_book.txt")

    def test_load_non_txt_file(self):
        """Test that non-txt files raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf',
                                          delete=False) as f:
            f.write("fake pdf content")
            f.flush()
            with self.assertRaises(ValueError):
                self.processor.load_text(f.name)
        os.unlink(f.name)

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello   world.\t\tThis   is   a   test.\n\n\n\nNew paragraph."
        result = self.processor.normalize_whitespace(text)
        self.assertNotIn('\t', result)
        self.assertNotIn('   ', result)  # No triple spaces

    def test_basic_stats(self):
        """Test basic statistics calculation."""
        text = "The quick brown fox. The lazy dog sleeps."
        stats = self.processor.get_basic_stats(text)
        self.assertGreater(stats['word_count'], 0)
        self.assertGreater(stats['sentence_count'], 0)
        self.assertGreater(stats['character_count'], 0)

    def test_clean_for_topic_modeling(self):
        """Test that cleaning removes non-alpha and lowercases."""
        text = "Hello World! Test 123."
        cleaned = self.processor.clean_for_topic_modeling(text)
        self.assertEqual(cleaned, "hello world test")

    def test_tokenize(self):
        """Test tokenization."""
        tokens = self.processor.tokenize("Hello world test")
        self.assertEqual(tokens, ["Hello", "world", "test"])

    def test_remove_stopwords(self):
        """Test stop word removal."""
        tokens = ["the", "quick", "brown", "fox", "is", "a"]
        filtered = self.processor.remove_stopwords(tokens)
        self.assertNotIn("the", filtered)
        self.assertNotIn("is", filtered)
        self.assertIn("quick", filtered)

    def test_lemmatize(self):
        """Test lemmatization."""
        tokens = ["running", "cats", "better"]
        result = self.processor.lemmatize(tokens)
        self.assertIn("running", result)  # Running lemmatizes to running (verb needs POS)
        self.assertIn("cat", result)

    def test_topic_modeling_tokens(self):
        """Test the full topic modeling token pipeline."""
        text = "The quick brown foxes jumped over the lazy dogs repeatedly."
        tokens = self.processor.get_tokens_for_topic_modeling(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        # Should not contain stop words
        for token in tokens:
            self.assertNotIn(token, {'the', 'over'})

    def test_detect_title(self):
        """Test title detection from filename."""
        title = self.processor.detect_title("data/books/my_great_book.txt", "")
        self.assertEqual(title, "My Great Book")

    def test_process_pipeline(self):
        """Test the full processing pipeline."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                          delete=False, encoding='utf-8') as f:
            f.write("The morning was quiet. The old man walked through the village. "
                    "He remembered the good old days clearly.")
            f.flush()
            result = self.processor.process(f.name)

            self.assertIn('raw_text', result)
            self.assertIn('normalized_text', result)
            self.assertIn('tokens_for_lda', result)
            self.assertIn('basic_stats', result)
            self.assertGreater(result['basic_stats']['word_count'], 0)
        os.unlink(f.name)

    def test_latin1_fallback(self):
        """Test that latin-1 encoded files can be read."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt',
                                          delete=False) as f:
            # Write latin-1 encoded text with special characters
            f.write("Héllo wörld. Café résumé.".encode('latin-1'))
            f.flush()
            text = self.processor.load_text(f.name)
            self.assertIn("llo", text)  # At least partial content
        os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()
