"""
E-Reader Text Analytics Engine - Visualization Module

Creates Matplotlib charts for sentiment arcs and topic distributions,
designed for embedding in Tkinter via FigureCanvasTkAgg.
"""

from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for embedding
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
import numpy as np


# Color palette
COLORS = {
    'primary': '#6366F1',      # Indigo
    'positive': '#10B981',     # Emerald
    'negative': '#EF4444',     # Red
    'neutral': '#8B5CF6',      # Purple
    'background': '#1E1E2E',   # Dark background
    'surface': '#2A2A3E',      # Card surface
    'text': '#E2E8F0',         # Light text
    'grid': '#374151',         # Grid lines
    'zero_line': '#9CA3AF',    # Zero reference
    'accent1': '#F59E0B',      # Amber
    'accent2': '#06B6D4',      # Cyan
}

TOPIC_COLORS = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#06B6D4',
                '#8B5CF6', '#EC4899', '#14B8A6', '#F97316', '#3B82F6']


def create_emotional_arc_figure(chapter_sentiments: List[Dict],
                                 figsize: tuple = (10, 5)) -> Figure:
    """Create an emotional arc line chart.

    Args:
        chapter_sentiments: List of dicts with 'title' and 'sentiment' keys.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure object ready for embedding.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['surface'])

    if not chapter_sentiments:
        ax.text(0.5, 0.5, 'No sentiment data available',
                ha='center', va='center', color=COLORS['text'], fontsize=14)
        return fig

    # Extract data
    chapters = [ch['title'] for ch in chapter_sentiments]
    # Shorten labels for display
    short_labels = []
    for ch in chapter_sentiments:
        label = f"Ch. {ch['number']}" if ch.get('number') else ch['title'][:15]
        short_labels.append(label)

    compounds = [ch['sentiment']['compound'] for ch in chapter_sentiments]
    positives = [ch['sentiment']['pos'] for ch in chapter_sentiments]
    negatives = [-ch['sentiment']['neg'] for ch in chapter_sentiments]  # Negative on y-axis

    x = range(len(compounds))

    # Plot compound sentiment line
    ax.plot(x, compounds, color=COLORS['primary'], linewidth=2.5,
            marker='o', markersize=8, markerfacecolor=COLORS['primary'],
            markeredgecolor='white', markeredgewidth=1.5,
            label='Compound Sentiment', zorder=3)

    # Fill areas above and below zero
    ax.fill_between(x, compounds, 0,
                     where=[c >= 0 for c in compounds],
                     alpha=0.15, color=COLORS['positive'], interpolate=True)
    ax.fill_between(x, compounds, 0,
                     where=[c < 0 for c in compounds],
                     alpha=0.15, color=COLORS['negative'], interpolate=True)

    # Zero reference line
    ax.axhline(y=0, color=COLORS['zero_line'], linewidth=1, linestyle='--', alpha=0.7)

    # Formatting
    ax.set_xlabel('Chapters', color=COLORS['text'], fontsize=11, labelpad=10)
    ax.set_ylabel('Sentiment Score', color=COLORS['text'], fontsize=11, labelpad=10)
    ax.set_title('Emotional Arc', color=COLORS['text'], fontsize=14,
                 fontweight='bold', pad=15)

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels, rotation=45, ha='right',
                        color=COLORS['text'], fontsize=9)
    ax.tick_params(axis='y', colors=COLORS['text'])

    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='--')
    ax.set_ylim(-1.05, 1.05)

    # Score annotations
    for i, score in enumerate(compounds):
        offset = 0.08 if score >= 0 else -0.12
        ax.annotate(f'{score:.2f}', (i, score + offset),
                    ha='center', va='bottom' if score >= 0 else 'top',
                    color=COLORS['text'], fontsize=8, fontweight='bold')

    ax.legend(loc='upper right', facecolor=COLORS['surface'],
              edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

    fig.tight_layout(pad=2)
    return fig


def create_topic_distribution_figure(topic_results: Dict,
                                      figsize: tuple = (10, 5)) -> Optional[Figure]:
    """Create a horizontal bar chart of topic keywords.

    Args:
        topic_results: Output from TopicModeler.build_model().
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure or None if no topics available.
    """
    if not topic_results.get('success') or not topic_results.get('topics'):
        return None

    topics = topic_results['topics']
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['surface'])

    labels = [f"Topic {t['topic_number']}" for t in topics]
    # Use the sum of keyword weights as a proxy for topic strength
    values = []
    keyword_texts = []
    for t in topics:
        weights = t.get('keyword_weights', [])
        total_weight = sum(w for _, w in weights) if weights else 0.1
        values.append(total_weight)
        keyword_texts.append(t['keywords'])

    y_pos = range(len(labels))
    colors = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(labels))]

    bars = ax.barh(y_pos, values, color=colors, height=0.6, alpha=0.85)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color=COLORS['text'], fontsize=10)
    ax.set_xlabel('Keyword Weight Sum', color=COLORS['text'], fontsize=10)
    ax.set_title('Discovered Topics', color=COLORS['text'],
                 fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='x', colors=COLORS['text'])

    # Add keyword annotations
    for i, (bar, keywords) in enumerate(zip(bars, keyword_texts)):
        # Truncate keywords for display
        display_kw = keywords[:60] + '...' if len(keywords) > 60 else keywords
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                display_kw, va='center', color=COLORS['text'],
                fontsize=8, alpha=0.8)

    ax.grid(True, axis='x', alpha=0.2, color=COLORS['grid'], linestyle='--')
    ax.invert_yaxis()

    fig.tight_layout(pad=2)
    return fig


def create_sentiment_breakdown_figure(chapter_sentiments: List[Dict],
                                       figsize: tuple = (10, 5)) -> Figure:
    """Create a stacked bar chart showing positive/neutral/negative breakdown.

    Args:
        chapter_sentiments: List of chapter sentiment dicts.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['surface'])

    if not chapter_sentiments:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                color=COLORS['text'], fontsize=14)
        return fig

    labels = [f"Ch. {ch['number']}" for ch in chapter_sentiments]
    pos_vals = [ch['sentiment']['pos'] for ch in chapter_sentiments]
    neu_vals = [ch['sentiment']['neu'] for ch in chapter_sentiments]
    neg_vals = [ch['sentiment']['neg'] for ch in chapter_sentiments]

    x = range(len(labels))
    width = 0.6

    ax.bar(x, pos_vals, width, label='Positive', color=COLORS['positive'], alpha=0.85)
    ax.bar(x, neu_vals, width, bottom=pos_vals, label='Neutral',
           color=COLORS['neutral'], alpha=0.85)
    bottoms = [p + n for p, n in zip(pos_vals, neu_vals)]
    ax.bar(x, neg_vals, width, bottom=bottoms, label='Negative',
           color=COLORS['negative'], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', color=COLORS['text'], fontsize=9)
    ax.set_ylabel('Proportion', color=COLORS['text'], fontsize=10)
    ax.set_title('Sentiment Breakdown by Chapter', color=COLORS['text'],
                 fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='y', colors=COLORS['text'])
    ax.legend(loc='upper right', facecolor=COLORS['surface'],
              edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    ax.grid(True, axis='y', alpha=0.2, color=COLORS['grid'], linestyle='--')

    fig.tight_layout(pad=2)
    return fig
