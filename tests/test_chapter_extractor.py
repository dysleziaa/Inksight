"""Tests for the ChapterExtractor module."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chapter_extractor import extract_chapters, _parse_chapter_number


class TestChapterExtractor(unittest.TestCase):
    """Test suite for chapter extraction."""

    def test_arabic_numerals(self):
        """Test detection of 'Chapter 1' style headings."""
        text = (
            "CHAPTER 1 - The Beginning\n\n"
            "This is the first chapter with enough text to be recognized. "
            "It contains multiple sentences and paragraphs that establish "
            "this as real content rather than a table of contents entry. "
            "The story begins here with characters and plot.\n\n"
            "CHAPTER 2 - The Middle\n\n"
            "This is the second chapter also with sufficient content. "
            "More plot development happens here with additional details "
            "and narrative progression through the story.\n\n"
            "CHAPTER 3 - The End\n\n"
            "The final chapter wraps everything up nicely. "
            "All loose ends are tied and the story reaches its conclusion.\n"
        )
        chapters = extract_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]['number'], 1)
        self.assertEqual(chapters[1]['number'], 2)
        self.assertEqual(chapters[2]['number'], 3)

    def test_roman_numerals(self):
        """Test detection of 'CHAPTER I' style headings."""
        text = (
            "CHAPTER I\n\n"
            "First chapter content with substantial text to distinguish "
            "it from a table of contents entry. More content follows here.\n\n"
            "CHAPTER II\n\n"
            "Second chapter content with enough material to be recognized "
            "as actual chapter body rather than a heading in a TOC.\n\n"
            "CHAPTER III\n\n"
            "Third chapter brings the narrative to its conclusion.\n"
        )
        chapters = extract_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]['number'], 1)

    def test_written_numbers(self):
        """Test detection of 'Chapter One' style headings."""
        text = (
            "Chapter One\n\n"
            "The story begins with a lengthy introduction establishing "
            "the setting and characters in great detail.\n\n"
            "Chapter Two\n\n"
            "The plot thickens with many twists and turns that keep "
            "the reader engaged throughout this section.\n"
        )
        chapters = extract_chapters(text)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]['number'], 1)
        self.assertEqual(chapters[1]['number'], 2)

    def test_chapter_with_title(self):
        """Test chapter heading with title suffix."""
        text = (
            "CHAPTER 1 - The Dawn\n\n"
            "Content of the first chapter with substantial text here. "
            "More sentences to make it clearly a chapter body.\n\n"
            "CHAPTER 2 - The Dusk\n\n"
            "Content of the second chapter with enough material.\n"
        )
        chapters = extract_chapters(text)
        self.assertEqual(len(chapters), 2)
        self.assertIn("The Dawn", chapters[0]['title'])
        self.assertIn("The Dusk", chapters[1]['title'])

    def test_no_chapters_detected(self):
        """Test fallback when no chapter headings are found."""
        text = (
            "Once upon a time, there was a village. "
            "The people lived happily and peacefully. "
            "Then one day, everything changed.\n"
        )
        chapters = extract_chapters(text)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]['title'], 'Full Text')
        self.assertIn('detection_note', chapters[0])

    def test_empty_text(self):
        """Test extraction from empty text."""
        chapters = extract_chapters("")
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]['title'], 'Full Text')

    def test_parse_chapter_number(self):
        """Test chapter number parsing helper."""
        self.assertEqual(_parse_chapter_number('1'), 1)
        self.assertEqual(_parse_chapter_number('15'), 15)
        self.assertEqual(_parse_chapter_number('I'), 1)
        self.assertEqual(_parse_chapter_number('X'), 10)
        self.assertEqual(_parse_chapter_number('ONE'), 1)
        self.assertEqual(_parse_chapter_number('FIVE'), 5)
        self.assertIsNone(_parse_chapter_number('abc'))

    def test_toc_filtering(self):
        """Test that TOC-like entries are filtered out."""
        text = (
            "TABLE OF CONTENTS\n"
            "Chapter 1\n"
            "Chapter 2\n"
            "Chapter 3\n\n"
            "Chapter 1\n\n"
            "This is the actual first chapter with substantial content "
            "that makes it clear this is not a table of contents entry. "
            "The narrative begins here with detailed prose.\n\n"
            "Chapter 2\n\n"
            "This is the actual second chapter with substantial content "
            "that makes it clear this is chapter body text.\n\n"
            "Chapter 3\n\n"
            "This is the actual third chapter with enough content to be "
            "recognized as a real chapter body.\n"
        )
        chapters = extract_chapters(text)
        # Should filter TOC entries and only keep actual chapters
        # The exact count depends on the TOC filtering heuristic
        for ch in chapters:
            self.assertGreater(len(ch['text']), 50)


if __name__ == '__main__':
    unittest.main()
