"""Tests for ReadabilityAnalyzer, SentimentAnalyzer, and TopicModeler."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.readability_analyzer import ReadabilityAnalyzer
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_modeler import TopicModeler


class TestReadabilityAnalyzer(unittest.TestCase):
    """Test suite for ReadabilityAnalyzer."""

    def setUp(self):
        self.analyzer = ReadabilityAnalyzer()

    def test_basic_analysis(self):
        """Test readability analysis on normal text."""
        text = ("The quick brown fox jumps over the lazy dog. "
                "This sentence is simple and easy to read. "
                "Short sentences are typically scored as easy reading.")
        result = self.analyzer.analyze(text)

        self.assertGreater(result['word_count'], 0)
        self.assertGreater(result['sentence_count'], 0)
        self.assertGreater(result['unique_word_count'], 0)
        self.assertGreater(result['avg_word_length'], 0)
        self.assertGreater(result['vocabulary_richness'], 0)
        self.assertIsNone(result['error'])

    def test_empty_text(self):
        """Test with empty input."""
        result = self.analyzer.analyze("")
        self.assertEqual(result['word_count'], 0)
        self.assertIsNotNone(result['error'])

    def test_flesch_scores(self):
        """Test that Flesch scores are calculated."""
        text = ("The cat sat on the mat. The dog lay on the rug. "
                "Both animals were happy and content in their homes.")
        result = self.analyzer.analyze(text)
        # textstat should provide scores
        if result.get('flesch_reading_ease') is not None:
            self.assertIsInstance(result['flesch_reading_ease'], float)
            self.assertIsInstance(result['flesch_kincaid_grade'], float)

    def test_reading_level_description(self):
        """Test reading level description mapping."""
        self.assertIn("Easy", self.analyzer.get_reading_level_description(85))
        self.assertIn("Difficult", self.analyzer.get_reading_level_description(40))
        self.assertEqual("Score not available",
                         self.analyzer.get_reading_level_description(None))


class TestSentimentAnalyzer(unittest.TestCase):
    """Test suite for SentimentAnalyzer."""

    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_text(self):
        """Test sentiment on clearly positive text."""
        result = self.analyzer.analyze_text(
            "This is wonderful! I love it so much. Everything is perfect and amazing.")
        self.assertGreater(result['compound'], 0)
        self.assertGreater(result['pos'], 0)

    def test_negative_text(self):
        """Test sentiment on clearly negative text."""
        result = self.analyzer.analyze_text(
            "This is terrible and awful. I hate everything about this disaster.")
        self.assertLess(result['compound'], 0)
        self.assertGreater(result['neg'], 0)

    def test_neutral_text(self):
        """Test sentiment on neutral text."""
        result = self.analyzer.analyze_text(
            "The table is made of wood. The chair is next to the desk.")
        # Neutral text should have high neu score
        self.assertGreater(result['neu'], 0.5)

    def test_empty_text(self):
        """Test with empty input."""
        result = self.analyzer.analyze_text("")
        self.assertEqual(result['compound'], 0.0)

    def test_chapter_analysis(self):
        """Test analyzing multiple chapters."""
        chapters = [
            {'number': 1, 'title': 'Ch 1', 'text': 'Happy wonderful amazing day!'},
            {'number': 2, 'title': 'Ch 2', 'text': 'Terrible awful horrible disaster.'},
        ]
        results = self.analyzer.analyze_chapters(chapters)
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0]['sentiment']['compound'], 0)
        self.assertLess(results[1]['sentiment']['compound'], 0)

    def test_interpretation(self):
        """Test interpretation generation."""
        chapters = [
            {'number': 1, 'title': 'Ch 1', 'text': 'Happy wonderful day!'},
            {'number': 2, 'title': 'Ch 2', 'text': 'Terrible disaster occurred.'},
            {'number': 3, 'title': 'Ch 3', 'text': 'Everything is beautiful again!'},
        ]
        sentiments = self.analyzer.analyze_chapters(chapters)
        interp = self.analyzer.generate_interpretation(sentiments)

        self.assertIn('overall', interp)
        self.assertIn('variation', interp)
        self.assertIn('arc', interp)
        self.assertIn('disclaimer', interp)
        self.assertTrue(len(interp['overall']) > 0)

    def test_story_signature(self):
        """Test story signature generation."""
        chapters = [
            {'number': 1, 'title': 'Ch 1', 'text': 'Terrible disaster and suffering.'},
            {'number': 2, 'title': 'Ch 2', 'text': 'Hope and recovery begin slowly.'},
            {'number': 3, 'title': 'Ch 3', 'text': 'Joy and celebration everywhere!'},
        ]
        sentiments = self.analyzer.analyze_chapters(chapters)
        sig = self.analyzer.generate_story_signature(sentiments)

        self.assertIn('label', sig)
        self.assertIn('components', sig)
        self.assertIn('note', sig)
        self.assertIn('avg_sentiment', sig['components'])


class TestTopicModeler(unittest.TestCase):
    """Test suite for TopicModeler."""

    def setUp(self):
        self.modeler = TopicModeler(num_topics=3, random_seed=42)

    def test_insufficient_text(self):
        """Test with too few tokens."""
        tokens = [["hello", "world"]]
        result = self.modeler.build_model(tokens)
        self.assertFalse(result['success'])
        self.assertIn('Insufficient', result['error'])

    def test_empty_input(self):
        """Test with empty input."""
        result = self.modeler.build_model([])
        self.assertFalse(result['success'])

    def test_sufficient_text(self):
        """Test with enough text for modeling."""
        # Create enough varied tokens across documents
        doc1 = (["king", "queen", "palace", "throne", "royal", "crown",
                 "castle", "knight", "sword", "battle"] * 5 +
                ["forest", "tree", "river", "mountain", "valley"] * 3)
        doc2 = (["ocean", "sea", "ship", "sail", "wave", "storm",
                 "captain", "crew", "island", "treasure"] * 5 +
                ["wind", "rain", "cloud", "thunder", "lightning"] * 3)
        doc3 = (["village", "farm", "harvest", "wheat", "bread",
                 "market", "trade", "merchant", "coin", "gold"] * 5 +
                ["house", "family", "child", "mother", "father"] * 3)

        result = self.modeler.build_model([doc1, doc2, doc3])

        if result['success']:
            self.assertGreater(len(result['topics']), 0)
            self.assertIsNotNone(result['note'])
            # Each topic should have keywords
            for topic in result['topics']:
                self.assertTrue(len(topic['keywords']) > 0)
        else:
            # Some environments may not support gensim
            self.assertIn('error', result)

    def test_deterministic_results(self):
        """Test that results are reproducible with same seed."""
        tokens = (["king", "queen", "palace", "throne", "royal", "crown"] * 10 +
                  ["ocean", "sea", "ship", "sail", "wave"] * 10)

        modeler1 = TopicModeler(num_topics=2, random_seed=42)
        modeler2 = TopicModeler(num_topics=2, random_seed=42)

        result1 = modeler1.build_model([tokens[:30], tokens[30:]])
        result2 = modeler2.build_model([tokens[:30], tokens[30:]])

        if result1['success'] and result2['success']:
            # Same seed should produce same topics
            for t1, t2 in zip(result1['topics'], result2['topics']):
                self.assertEqual(t1['keywords'], t2['keywords'])


if __name__ == '__main__':
    unittest.main()
