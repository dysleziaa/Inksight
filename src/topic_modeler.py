"""
E-Reader Text Analytics Engine - Topic Modeler Module

Implements LDA (Latent Dirichlet Allocation) topic modeling using Gensim.
Discovers word clusters that may represent themes in the text.
"""

import logging
from typing import List, Dict, Optional, Tuple

try:
    from gensim import corpora, models
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Suppress excessive gensim logging
logging.getLogger('gensim').setLevel(logging.WARNING)


# Minimum thresholds for meaningful LDA
MIN_DOCUMENTS = 2
MIN_UNIQUE_WORDS = 20
MIN_TOTAL_TOKENS = 50


class TopicModeler:
    """LDA topic modeling for book text analysis."""

    def __init__(self, num_topics: int = 5, random_seed: int = 42):
        """Initialize the topic modeler.

        Args:
            num_topics: Number of topics to discover (default 5).
            random_seed: Random seed for reproducibility.
        """
        self.num_topics = num_topics
        self.random_seed = random_seed
        self.model = None
        self.dictionary = None
        self.corpus = None

    def build_model(self, chapter_tokens: List[List[str]]) -> Dict[str, object]:
        """Build an LDA model from tokenized chapter documents.

        Args:
            chapter_tokens: List of token lists, one per chapter/document.

        Returns:
            Dictionary with:
                - success: Whether model was built
                - topics: List of (topic_num, keywords) tuples
                - topic_distributions: Per-document topic distributions
                - error: Error message if any
                - note: Interpretive note
        """
        if not GENSIM_AVAILABLE:
            return {
                'success': False,
                'topics': [],
                'topic_distributions': [],
                'error': 'Gensim library not installed. Install with: pip install gensim',
                'note': '',
            }

        # Validate input sufficiency
        total_tokens = sum(len(doc) for doc in chapter_tokens)
        all_tokens_flat = [t for doc in chapter_tokens for t in doc]
        unique_words = set(all_tokens_flat)

        if total_tokens < MIN_TOTAL_TOKENS:
            return {
                'success': False,
                'topics': [],
                'topic_distributions': [],
                'error': (f"Insufficient text for topic modeling. "
                          f"Found {total_tokens} tokens, need at least {MIN_TOTAL_TOKENS}."),
                'note': '',
            }

        if len(unique_words) < MIN_UNIQUE_WORDS:
            return {
                'success': False,
                'topics': [],
                'topic_distributions': [],
                'error': (f"Insufficient vocabulary for topic modeling. "
                          f"Found {len(unique_words)} unique words, need at least {MIN_UNIQUE_WORDS}."),
                'note': '',
            }

        # If only one document, split into paragraphs or chunks for better modeling
        if len(chapter_tokens) < MIN_DOCUMENTS:
            chapter_tokens = self._split_into_chunks(all_tokens_flat, chunk_size=100)
            if len(chapter_tokens) < MIN_DOCUMENTS:
                return {
                    'success': False,
                    'topics': [],
                    'topic_distributions': [],
                    'error': "Text too short to create sufficient documents for topic modeling.",
                    'note': '',
                }

        # Adjust num_topics if we have very few documents
        effective_topics = min(self.num_topics, len(chapter_tokens), len(unique_words) // 5)
        effective_topics = max(effective_topics, 2)

        try:
            # Build dictionary and corpus
            self.dictionary = corpora.Dictionary(chapter_tokens)

            # Filter extremes to remove very rare and very common words
            if len(self.dictionary) > 50:
                self.dictionary.filter_extremes(no_below=1, no_above=0.9)

            self.corpus = [self.dictionary.doc2bow(doc) for doc in chapter_tokens]

            # Build LDA model
            self.model = models.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=effective_topics,
                random_state=self.random_seed,
                passes=15,
                alpha='auto',
                per_word_topics=False,
            )

            # Extract topics
            topics = []
            for topic_id in range(effective_topics):
                keywords = self.model.show_topic(topic_id, topn=8)
                keyword_str = ", ".join([word for word, prob in keywords])
                topics.append({
                    'topic_number': topic_id + 1,
                    'keywords': keyword_str,
                    'keyword_weights': [(word, round(prob, 4)) for word, prob in keywords],
                })

            # Get per-document topic distributions
            topic_distributions = []
            for i, bow in enumerate(self.corpus):
                doc_topics = self.model.get_document_topics(bow, minimum_probability=0.01)
                dist = {f"Topic {tid + 1}": round(prob, 4) for tid, prob in doc_topics}
                topic_distributions.append(dist)

            note = ("These topics are estimated word clusters discovered by the LDA algorithm. "
                    "They require human interpretation to determine if they correspond to "
                    "meaningful themes in the text.")

            return {
                'success': True,
                'topics': topics,
                'topic_distributions': topic_distributions,
                'num_topics': effective_topics,
                'error': None,
                'note': note,
            }

        except Exception as e:
            return {
                'success': False,
                'topics': [],
                'topic_distributions': [],
                'error': f"Topic modeling failed: {str(e)}",
                'note': '',
            }

    def get_chapter_topic_proportions(self, chapters: List[Dict],
                                       topic_results: Dict) -> List[Dict]:
        """Map topic distributions back to chapter names.

        Args:
            chapters: Chapter dicts from chapter_extractor.
            topic_results: Output of build_model.

        Returns:
            List of dicts mapping chapter titles to their topic proportions.
        """
        if not topic_results.get('success') or not topic_results.get('topic_distributions'):
            return []

        distributions = topic_results['topic_distributions']
        result = []

        for i, ch in enumerate(chapters):
            if i < len(distributions):
                result.append({
                    'chapter': ch['title'],
                    'topics': distributions[i],
                })

        return result

    def _split_into_chunks(self, tokens: List[str], chunk_size: int = 100) -> List[List[str]]:
        """Split a flat token list into chunks for modeling."""
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) >= 20:  # Only keep chunks with enough tokens
                chunks.append(chunk)
        return chunks
