"""
E-Reader Text Analytics Engine - Database Module

Handles MySQL database connections, table creation, and CRUD operations
for storing book analysis results. Designed to fail gracefully when
MySQL is unavailable.
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    MySQLError = Exception

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; environment variables can be set externally


# Default database configuration
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'port': int(os.environ.get('MYSQL_PORT', '3306')),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', ''),
    'database': 'ereader_analytics',
}

CREATE_BOOKS_TABLE = """
CREATE TABLE IF NOT EXISTS books (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(255) DEFAULT NULL,
    filename VARCHAR(500) NOT NULL,
    word_count INT DEFAULT 0,
    sentence_count INT DEFAULT 0,
    chapter_count INT DEFAULT 0,
    readability_score FLOAT DEFAULT NULL,
    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_CHAPTER_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS chapter_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    book_id INT NOT NULL,
    chapter_number INT NOT NULL,
    chapter_title VARCHAR(500) DEFAULT '',
    word_count INT DEFAULT 0,
    sentiment_compound FLOAT DEFAULT 0,
    sentiment_positive FLOAT DEFAULT 0,
    sentiment_neutral FLOAT DEFAULT 0,
    sentiment_negative FLOAT DEFAULT 0,
    FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_TOPICS_TABLE = """
CREATE TABLE IF NOT EXISTS topics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    book_id INT NOT NULL,
    topic_number INT NOT NULL,
    keywords TEXT NOT NULL,
    FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


class DatabaseManager:
    """Manages MySQL database operations for the analytics engine.

    All methods handle connection failures gracefully and return
    status information so the UI can inform the user.
    """

    def __init__(self):
        self._connection = None
        self._available = False
        self._last_error = None

    @property
    def is_available(self) -> bool:
        """Check if database connection is available."""
        return self._available

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error

    def connect(self) -> bool:
        """Establish database connection and create tables.

        Returns:
            True if connection succeeded, False otherwise.
        """
        if not MYSQL_AVAILABLE:
            self._last_error = ("mysql-connector-python is not installed. "
                                "Install with: pip install mysql-connector-python")
            self._available = False
            return False

        try:
            # First connect without database to create it if needed
            config_no_db = {k: v for k, v in DB_CONFIG.items() if k != 'database'}
            conn = mysql.connector.connect(**config_no_db)
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.close()
            conn.close()

            # Now connect to the database
            self._connection = mysql.connector.connect(**DB_CONFIG)
            self._create_tables()
            self._available = True
            self._last_error = None
            return True

        except Exception as e:
            self._last_error = f"Database connection failed: {str(e)}"
            self._available = False
            return False

    def _create_tables(self):
        """Create required tables if they don't exist."""
        cursor = self._connection.cursor()
        for sql in [CREATE_BOOKS_TABLE, CREATE_CHAPTER_ANALYSIS_TABLE, CREATE_TOPICS_TABLE]:
            cursor.execute(sql)
        self._connection.commit()
        cursor.close()

    def _ensure_connection(self) -> bool:
        """Ensure connection is alive, attempt reconnect if needed."""
        if not self._available or self._connection is None:
            return False
        try:
            self._connection.ping(reconnect=True, attempts=2, delay=1)
            return True
        except Exception:
            self._available = False
            self._last_error = "Database connection lost."
            return False

    def save_analysis(self, book_data: Dict, chapters: List[Dict],
                       chapter_sentiments: List[Dict],
                       topic_results: Optional[Dict],
                       readability: Dict) -> Tuple[bool, str]:
        """Save a complete analysis to the database.

        Args:
            book_data: Dict with title, file_path, basic_stats.
            chapters: List of chapter dicts.
            chapter_sentiments: List of sentiment results per chapter.
            topic_results: Topic modeling results (may be None).
            readability: Readability metrics dict.

        Returns:
            Tuple of (success: bool, message: str).
        """
        if not self._ensure_connection():
            return False, f"Cannot save: {self._last_error or 'No database connection'}"

        try:
            cursor = self._connection.cursor()

            # Insert book record
            cursor.execute(
                """INSERT INTO books (title, filename, word_count, sentence_count,
                   chapter_count, readability_score, analyzed_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    book_data.get('title', 'Unknown'),
                    os.path.basename(book_data.get('file_path', '')),
                    readability.get('word_count', 0),
                    readability.get('sentence_count', 0),
                    len(chapters),
                    readability.get('flesch_reading_ease'),
                    datetime.now(),
                )
            )
            book_id = cursor.lastrowid

            # Insert chapter analysis records
            for cs in chapter_sentiments:
                # Find matching chapter for word count
                ch_word_count = 0
                for ch in chapters:
                    if ch['number'] == cs['number']:
                        ch_word_count = len(ch['text'].split())
                        break

                sentiment = cs['sentiment']
                cursor.execute(
                    """INSERT INTO chapter_analysis
                       (book_id, chapter_number, chapter_title, word_count,
                        sentiment_compound, sentiment_positive,
                        sentiment_neutral, sentiment_negative)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        book_id,
                        cs['number'],
                        cs['title'],
                        ch_word_count,
                        sentiment.get('compound', 0),
                        sentiment.get('pos', 0),
                        sentiment.get('neu', 0),
                        sentiment.get('neg', 0),
                    )
                )

            # Insert topic records
            if topic_results and topic_results.get('success'):
                for topic in topic_results.get('topics', []):
                    cursor.execute(
                        """INSERT INTO topics (book_id, topic_number, keywords)
                           VALUES (%s, %s, %s)""",
                        (book_id, topic['topic_number'], topic['keywords'])
                    )

            self._connection.commit()
            cursor.close()
            return True, f"Analysis saved successfully (Book ID: {book_id})"

        except Exception as e:
            try:
                self._connection.rollback()
            except Exception:
                pass
            self._last_error = str(e)
            return False, f"Failed to save analysis: {str(e)}"

    def get_history(self) -> Tuple[bool, List[Dict]]:
        """Retrieve list of previously analyzed books.

        Returns:
            Tuple of (success, list of book records).
        """
        if not self._ensure_connection():
            return False, []

        try:
            cursor = self._connection.cursor(dictionary=True)
            cursor.execute(
                """SELECT id, title, filename, word_count, sentence_count,
                   chapter_count, readability_score, analyzed_at
                   FROM books ORDER BY analyzed_at DESC"""
            )
            rows = cursor.fetchall()
            cursor.close()
            return True, rows

        except Exception as e:
            self._last_error = str(e)
            return False, []

    def get_book_details(self, book_id: int) -> Tuple[bool, Optional[Dict]]:
        """Retrieve detailed analysis for a specific book.

        Args:
            book_id: The database ID of the book.

        Returns:
            Tuple of (success, details dict or None).
        """
        if not self._ensure_connection():
            return False, None

        try:
            cursor = self._connection.cursor(dictionary=True)

            # Book info
            cursor.execute("SELECT * FROM books WHERE id = %s", (book_id,))
            book = cursor.fetchone()
            if not book:
                cursor.close()
                return False, None

            # Chapter sentiments
            cursor.execute(
                """SELECT chapter_number, chapter_title, word_count,
                   sentiment_compound, sentiment_positive,
                   sentiment_neutral, sentiment_negative
                   FROM chapter_analysis WHERE book_id = %s
                   ORDER BY chapter_number""",
                (book_id,)
            )
            chapters = cursor.fetchall()

            # Topics
            cursor.execute(
                "SELECT topic_number, keywords FROM topics WHERE book_id = %s ORDER BY topic_number",
                (book_id,)
            )
            topics = cursor.fetchall()

            cursor.close()
            return True, {
                'book': book,
                'chapters': chapters,
                'topics': topics,
            }

        except Exception as e:
            self._last_error = str(e)
            return False, None

    def delete_book(self, book_id: int) -> Tuple[bool, str]:
        """Delete a book and its analysis from the database.

        Args:
            book_id: The database ID of the book to delete.

        Returns:
            Tuple of (success, message).
        """
        if not self._ensure_connection():
            return False, f"Cannot delete: {self._last_error or 'No database connection'}"

        try:
            cursor = self._connection.cursor()
            cursor.execute("DELETE FROM books WHERE id = %s", (book_id,))
            self._connection.commit()
            cursor.close()
            return True, "Book deleted successfully."
        except Exception as e:
            self._last_error = str(e)
            return False, f"Failed to delete book: {str(e)}"

    def close(self):
        """Close the database connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
            self._available = False
