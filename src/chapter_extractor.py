"""
E-Reader Text Analytics Engine - Chapter Extractor Module

Detects and extracts chapters from book text using common heading patterns.
Handles Roman numerals, written-out numbers, and various formatting styles.
"""

import re
from typing import List, Dict, Optional


# Roman numeral mapping for chapter detection
ROMAN_NUMERALS = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
    'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25,
    'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30,
}

# Written-out number mapping
WRITTEN_NUMBERS = {
    'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
    'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9, 'TEN': 10,
    'ELEVEN': 11, 'TWELVE': 12, 'THIRTEEN': 13, 'FOURTEEN': 14,
    'FIFTEEN': 15, 'SIXTEEN': 16, 'SEVENTEEN': 17, 'EIGHTEEN': 18,
    'NINETEEN': 19, 'TWENTY': 20,
}

# Chapter heading patterns (compiled regex)
# Matches lines like: "Chapter 1", "CHAPTER I", "Chapter One", "CHAPTER 1 - Title"
CHAPTER_PATTERN = re.compile(
    r'^\s*'                                # Optional leading whitespace
    r'(?:CHAPTER|Chapter|chapter)'         # "Chapter" in various cases
    r'\s+'                                 # Required whitespace
    r'('
        r'\d+'                             # Arabic numeral
        r'|[IVXLC]+'                       # Roman numeral
        r'|(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten'
        r'|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen'
        r'|Seventeen|Eighteen|Nineteen|Twenty'
        r'|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN'
        r'|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN'
        r'|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY)'  # Written numbers
    r')'
    r'(?:'                                 # Optional title suffix
        r'\s*[-:.]\s*'                     # Separator
        r'(.+)'                            # Title text
    r')?'
    r'\s*$',                               # Trailing whitespace
    re.MULTILINE
)


def _is_toc_context(text: str, match_start: int, match_end: int,
                     min_body_chars: int = 50) -> bool:
    """Check if a chapter heading appears to be in a table of contents.

    A TOC entry typically has very little text before the next chapter heading.
    We look at whether there's substantial body text after the heading before
    the next heading appears.

    Args:
        text: The full book text.
        match_start: Start position of the current match.
        match_end: End position of the current match (after the heading line).
        min_body_chars: Minimum characters expected between chapter headings
                        in actual content (vs. TOC entries).

    Returns:
        True if this appears to be a TOC entry rather than actual chapter start.
    """
    # Find next chapter heading starting after the current match ends
    next_match = CHAPTER_PATTERN.search(text, match_end)
    if next_match is None:
        return False

    # Check text between end of this heading and start of the next
    between_text = text[match_end:next_match.start()].strip()
    # If very little text between headings, likely a TOC
    return len(between_text) < min_body_chars


def _parse_chapter_number(number_str: str) -> Optional[int]:
    """Parse a chapter number from string (arabic, roman, or written).

    Args:
        number_str: The raw number string from the regex match.

    Returns:
        Integer chapter number or None if unparseable.
    """
    # Try arabic numeral
    try:
        return int(number_str)
    except ValueError:
        pass

    # Try roman numeral
    upper = number_str.upper()
    if upper in ROMAN_NUMERALS:
        return ROMAN_NUMERALS[upper]

    # Try written number
    if upper in WRITTEN_NUMBERS:
        return WRITTEN_NUMBERS[upper]

    return None


def extract_chapters(text: str) -> List[Dict[str, str]]:
    """Extract chapters from book text.

    Detects chapter headings using common patterns and splits the text
    accordingly. Filters out likely table-of-contents entries.

    Args:
        text: The full normalized text of the book.

    Returns:
        List of dicts, each with:
            - 'number': Chapter number (int)
            - 'title': Chapter title/heading (str)
            - 'text': Full text of the chapter (str)
            - 'heading': The original heading line (str)

        If no chapters are detected, returns a single entry containing
        the entire text as "Full Text".
    """
    matches = list(CHAPTER_PATTERN.finditer(text))

    if not matches:
        return [{
            'number': 1,
            'title': 'Full Text',
            'text': text.strip(),
            'heading': 'Full Text',
            'detection_note': 'No chapter headings detected. Treating entire book as one section.'
        }]

    # Filter out TOC entries
    content_matches = []
    for match in matches:
        if not _is_toc_context(text, match.start(), match.end()):
            content_matches.append(match)

    if not content_matches:
        return [{
            'number': 1,
            'title': 'Full Text',
            'text': text.strip(),
            'heading': 'Full Text',
            'detection_note': 'Chapter headings found only in table of contents. Treating entire book as one section.'
        }]

    chapters = []
    for i, match in enumerate(content_matches):
        number_str = match.group(1)
        title_str = match.group(2) if match.group(2) else ''
        chapter_num = _parse_chapter_number(number_str) or (i + 1)
        heading = match.group(0).strip()

        # Chapter text runs from this heading to the next (or end of text)
        start = match.start()
        if i + 1 < len(content_matches):
            end = content_matches[i + 1].start()
        else:
            end = len(text)

        chapter_text = text[start:end].strip()

        # Build display title
        if title_str:
            display_title = f"Chapter {chapter_num} - {title_str.strip()}"
        else:
            display_title = f"Chapter {chapter_num}"

        chapters.append({
            'number': chapter_num,
            'title': display_title,
            'text': chapter_text,
            'heading': heading,
        })

    return chapters
