"""
E-Reader Text Analytics Engine - Report Exporter Module

Exports analysis results to CSV or JSON format.
"""

import csv
import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class ReportExporter:
    """Exports analysis results to CSV or JSON files."""

    def export_json(self, filepath: str, book_data: Dict, readability: Dict,
                     chapters: List[Dict], chapter_sentiments: List[Dict],
                     topic_results: Optional[Dict],
                     story_signature: Optional[Dict],
                     interpretation: Optional[Dict]) -> str:
        """Export full analysis report to JSON.

        Args:
            filepath: Output file path.
            book_data: Book metadata dict.
            readability: Readability metrics.
            chapters: Chapter list.
            chapter_sentiments: Sentiment results per chapter.
            topic_results: Topic modeling results.
            story_signature: Story signature data.
            interpretation: Sentiment interpretation.

        Returns:
            Path to the created file.
        """
        report = {
            'report_generated': datetime.now().isoformat(),
            'book': {
                'title': book_data.get('title', 'Unknown'),
                'filename': os.path.basename(book_data.get('file_path', '')),
                'chapter_count': len(chapters),
            },
            'readability': readability,
            'chapters': [],
            'topics': [],
            'story_signature': story_signature or {},
            'interpretation': interpretation or {},
        }

        # Chapter details
        for cs in chapter_sentiments:
            ch_data = {
                'number': cs['number'],
                'title': cs['title'],
                'sentiment': cs['sentiment'],
            }
            # Add word count from chapter
            for ch in chapters:
                if ch['number'] == cs['number']:
                    ch_data['word_count'] = len(ch['text'].split())
                    break
            report['chapters'].append(ch_data)

        # Topics
        if topic_results and topic_results.get('success'):
            for t in topic_results.get('topics', []):
                report['topics'].append({
                    'topic_number': t['topic_number'],
                    'keywords': t['keywords'],
                })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return filepath

    def export_csv(self, filepath: str, book_data: Dict, readability: Dict,
                    chapters: List[Dict], chapter_sentiments: List[Dict],
                    topic_results: Optional[Dict]) -> str:
        """Export analysis report to CSV.

        Creates a CSV with chapter-level data and book-level summary.

        Args:
            filepath: Output file path.
            book_data: Book metadata.
            readability: Readability metrics.
            chapters: Chapter list.
            chapter_sentiments: Sentiment results.
            topic_results: Topic modeling results.

        Returns:
            Path to the created file.
        """
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Book summary section
            writer.writerow(['=== BOOK SUMMARY ==='])
            writer.writerow(['Title', book_data.get('title', 'Unknown')])
            writer.writerow(['Filename', os.path.basename(book_data.get('file_path', ''))])
            writer.writerow(['Total Words', readability.get('word_count', 0)])
            writer.writerow(['Total Sentences', readability.get('sentence_count', 0)])
            writer.writerow(['Unique Words', readability.get('unique_word_count', 0)])
            writer.writerow(['Avg Words/Sentence', readability.get('avg_words_per_sentence', 0)])
            writer.writerow(['Avg Word Length', readability.get('avg_word_length', 0)])
            writer.writerow(['Vocabulary Richness', readability.get('vocabulary_richness', 0)])
            writer.writerow(['Flesch Reading Ease', readability.get('flesch_reading_ease', 'N/A')])
            writer.writerow(['Flesch-Kincaid Grade', readability.get('flesch_kincaid_grade', 'N/A')])
            writer.writerow([])

            # Chapter sentiment section
            writer.writerow(['=== CHAPTER SENTIMENT ==='])
            writer.writerow(['Chapter', 'Title', 'Compound', 'Positive', 'Neutral', 'Negative'])
            for cs in chapter_sentiments:
                s = cs['sentiment']
                writer.writerow([
                    cs['number'], cs['title'],
                    s.get('compound', 0), s.get('pos', 0),
                    s.get('neu', 0), s.get('neg', 0),
                ])
            writer.writerow([])

            # Topics section
            if topic_results and topic_results.get('success'):
                writer.writerow(['=== TOPICS ==='])
                writer.writerow(['Topic Number', 'Keywords'])
                for t in topic_results.get('topics', []):
                    writer.writerow([t['topic_number'], t['keywords']])

        return filepath
