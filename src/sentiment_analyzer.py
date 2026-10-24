"""
E-Reader Text Analytics Engine - Sentiment Analyzer Module

Uses VADER Sentiment Analysis for chapter-level sentiment scoring
and generates emotional arc interpretations.
"""

from typing import List, Dict, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class SentimentAnalyzer:
    """Performs VADER sentiment analysis on chapter text."""

    def __init__(self):
        if VADER_AVAILABLE:
            self._analyzer = SentimentIntensityAnalyzer()
        else:
            self._analyzer = None

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text passage.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with compound, pos, neu, neg scores.
        """
        if not VADER_AVAILABLE:
            return {
                'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0,
                'error': 'vaderSentiment not installed.'
            }

        if not text or not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}

        scores = self._analyzer.polarity_scores(text)
        return {
            'compound': round(scores['compound'], 4),
            'pos': round(scores['pos'], 4),
            'neu': round(scores['neu'], 4),
            'neg': round(scores['neg'], 4),
        }

    def analyze_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """Analyze sentiment for each chapter.

        Args:
            chapters: List of chapter dicts (from chapter_extractor),
                      each must have 'text' and 'title' keys.

        Returns:
            List of dicts with chapter info and sentiment scores.
        """
        results = []
        for ch in chapters:
            sentiment = self.analyze_text(ch['text'])
            results.append({
                'number': ch['number'],
                'title': ch['title'],
                'sentiment': sentiment,
            })
        return results

    def generate_interpretation(self, chapter_sentiments: List[Dict]) -> Dict[str, str]:
        """Generate a text interpretation of the emotional arc.

        Args:
            chapter_sentiments: Output of analyze_chapters.

        Returns:
            Dictionary with interpretation fields:
                - overall: Overall sentiment tendency description
                - variation: Whether sentiment varies across chapters
                - arc: Comparison of opening vs ending
                - disclaimer: Methodological note
        """
        if not chapter_sentiments:
            return {
                'overall': 'No data available.',
                'variation': '',
                'arc': '',
                'disclaimer': '',
            }

        compounds = [ch['sentiment']['compound'] for ch in chapter_sentiments]
        avg_compound = sum(compounds) / len(compounds)
        min_compound = min(compounds)
        max_compound = max(compounds)
        spread = max_compound - min_compound

        # Overall tendency
        if avg_compound >= 0.15:
            overall = f"The overall sentiment tends positive (avg: {avg_compound:.3f})."
        elif avg_compound <= -0.15:
            overall = f"The overall sentiment tends negative (avg: {avg_compound:.3f})."
        else:
            overall = f"The overall sentiment is mixed/neutral (avg: {avg_compound:.3f})."

        # Variation
        if len(compounds) > 1:
            if spread >= 0.3:
                variation = (f"Sentiment varies substantially across chapters "
                             f"(range: {min_compound:.3f} to {max_compound:.3f}).")
            else:
                variation = (f"Sentiment is relatively consistent across chapters "
                             f"(range: {min_compound:.3f} to {max_compound:.3f}).")
        else:
            variation = "Only one section analyzed; no variation to measure."

        # Arc (opening vs ending)
        if len(compounds) >= 2:
            opening = compounds[0]
            ending = compounds[-1]
            diff = ending - opening
            if diff > 0.1:
                arc = (f"The ending ({ending:.3f}) is more positive than "
                       f"the opening ({opening:.3f}), suggesting an upward arc.")
            elif diff < -0.1:
                arc = (f"The ending ({ending:.3f}) is more negative than "
                       f"the opening ({opening:.3f}), suggesting a downward arc.")
            else:
                arc = (f"The opening ({opening:.3f}) and ending ({ending:.3f}) "
                       f"have similar sentiment.")
        else:
            arc = ""

        disclaimer = ("Note: VADER sentiment analysis measures word-level emotional "
                      "valence. It does not fully understand literary themes, irony, "
                      "or narrative context.")

        return {
            'overall': overall,
            'variation': variation,
            'arc': arc,
            'disclaimer': disclaimer,
        }

    def generate_story_signature(self, chapter_sentiments: List[Dict]) -> Dict[str, object]:
        """Generate an experimental story signature from sentiment data.

        Uses transparent, rule-based heuristics:
        - Average compound sentiment → positive/negative/mixed tendency
        - Sentiment variability (std dev) → stable/dynamic
        - Opening vs ending difference → upward/downward/flat arc
        - Final chapter sentiment → ending tone

        Args:
            chapter_sentiments: Output of analyze_chapters.

        Returns:
            Dictionary with signature components and optional labels.
        """
        if not chapter_sentiments:
            return {'label': 'Insufficient data', 'components': {}, 'note': ''}

        compounds = [ch['sentiment']['compound'] for ch in chapter_sentiments]
        avg = sum(compounds) / len(compounds)

        # Standard deviation
        if len(compounds) > 1:
            variance = sum((c - avg) ** 2 for c in compounds) / len(compounds)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        final = compounds[-1]
        opening = compounds[0]
        arc_diff = final - opening

        components = {
            'avg_sentiment': round(avg, 4),
            'sentiment_variability': round(std_dev, 4),
            'arc_direction': round(arc_diff, 4),
            'final_sentiment': round(final, 4),
        }

        # Determine label using documented rules
        labels = []
        if avg < -0.15:
            labels.append("negative tendency")
        elif avg > 0.15:
            labels.append("positive tendency")
        else:
            labels.append("mixed sentiment")

        if std_dev > 0.15:
            labels.append("emotionally dynamic")
        else:
            labels.append("emotionally stable")

        if len(compounds) >= 2:
            if arc_diff > 0.15:
                labels.append("uplifting-like arc")
            elif arc_diff < -0.15:
                labels.append("tragic-like arc")
            else:
                labels.append("flat arc")

        signature_label = " | ".join(labels)

        note = ("This is an exploratory statistical interpretation based on word-level "
                "sentiment scoring. It is not a scientifically validated classification "
                "of the book's genre or narrative meaning.")

        return {
            'label': signature_label,
            'components': components,
            'note': note,
        }
