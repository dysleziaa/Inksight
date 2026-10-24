# E-Reader Text Analytics Engine

A desktop NLP application that analyzes text-based books for readability, sentiment, topic themes, and emotional arcs. Built with Python, Tkinter, NLTK, and Gensim.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 📋 Project Overview

### Problem Statement

Readers, students, and researchers often want to understand the linguistic patterns, emotional trajectories, and thematic composition of books — beyond just reading them. Manual analysis is time-consuming and subjective.

### Objectives

- **Automate text analysis** of plain-text books using established NLP techniques.
- **Visualize emotional arcs** showing how sentiment changes chapter by chapter.
- **Discover latent topics** using unsupervised machine learning (LDA).
- **Calculate readability metrics** to assess text complexity.
- **Store and compare** analyses across multiple books.

### How It Works

1. **Upload** a `.txt` book file.
2. **Process** the text: tokenization, stop-word removal, lemmatization.
3. **Extract chapters** using pattern-matching on common heading formats.
4. **Analyze** readability (Flesch scores), sentiment (VADER), and topics (LDA).
5. **Visualize** results in an interactive dashboard with charts.
6. **Save** results to MySQL and export reports to CSV/JSON.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Tkinter Desktop UI                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Overview  │ │Readability│ │  Topics  │ │Sentiment │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
├──────────────────────────────────────────────────────────────┤
│                   Analysis Pipeline                          │
│  TextProcessor → ChapterExtractor → ReadabilityAnalyzer     │
│                → SentimentAnalyzer → TopicModeler            │
├──────────────────────────────────────────────────────────────┤
│  Visualization (Matplotlib)  │  Database (MySQL)             │
│  Report Exporter (CSV/JSON)  │  Reading History              │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
.txt File → TextProcessor (load, normalize, tokenize)
         → ChapterExtractor (detect chapter boundaries)
         → ReadabilityAnalyzer (Flesch scores, word stats)
         → SentimentAnalyzer (VADER per chapter → emotional arc)
         → TopicModeler (LDA topic discovery)
         → Visualization (Matplotlib charts embedded in Tkinter)
         → DatabaseManager (MySQL persistence)
         → ReportExporter (CSV/JSON output)
```

---

## 🛠️ Technology Stack

| Technology | Purpose | Why |
|---|---|---|
| **Python 3.12** | Core language | Rich NLP ecosystem, widely taught |
| **Tkinter + ttk** | Desktop UI | Built into Python, no extra dependencies |
| **NLTK** | Text preprocessing | Industry-standard NLP toolkit for tokenization, lemmatization, stop words |
| **textstat** | Readability scores | Implements Flesch Reading Ease and Flesch-Kincaid Grade Level formulas |
| **Gensim** | Topic modeling (LDA) | Efficient, well-documented LDA implementation |
| **VADER Sentiment** | Sentiment analysis | Designed for social media/text sentiment; works well on literary prose |
| **Matplotlib** | Data visualization | Flexible charting with Tkinter embedding support |
| **MySQL** | Data persistence | Relational storage for structured analysis results |
| **mysql-connector-python** | DB connector | Official MySQL Python driver |
| **python-dotenv** | Configuration | Secure credential management via `.env` files |

---

## 📁 Project Structure

```
E-reader/
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── .env                     # MySQL credentials (not committed)
├── .env.example             # Template for .env
├── .gitignore
├── README.md
├── data/
│   └── books/               # Place .txt book files here
│       ├── sample.txt
│       └── test_chapters.txt
├── src/
│   ├── __init__.py
│   ├── text_processor.py    # Text loading, cleaning, tokenization
│   ├── chapter_extractor.py # Chapter heading detection
│   ├── readability_analyzer.py  # Flesch scores, word statistics
│   ├── sentiment_analyzer.py    # VADER sentiment + story signature
│   ├── topic_modeler.py     # Gensim LDA topic modeling
│   ├── visualization.py     # Matplotlib chart generation
│   ├── database.py          # MySQL database operations
│   ├── report_exporter.py   # CSV/JSON export
│   └── ui.py                # Tkinter desktop interface
└── tests/
    ├── test_text_processor.py
    ├── test_chapter_extractor.py
    └── test_analysis.py
```

---

## 🚀 Installation (Windows)

### Prerequisites

- **Python 3.10+** installed ([python.org](https://python.org))
- **MySQL Server 8.0+** installed and running ([dev.mysql.com](https://dev.mysql.com/downloads/mysql/))

### Step 1: Clone or Download the Project

```powershell
cd D:\
git clone <repository-url> E-reader
cd E-reader
```

### Step 2: (Optional) Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Configure MySQL

1. **Start MySQL** (it may already be running as a Windows service).

2. **Create the `.env` file** from the template:
   ```powershell
   copy .env.example .env
   ```

3. **Edit `.env`** with your MySQL password:
   ```
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_password
   ```

4. The database and tables are **created automatically** when the application starts.

> **Note:** The application works without MySQL — analysis results just won't be saved to the database.

### Step 5: Download NLTK Data (automatic)

NLTK data packages are downloaded automatically on first run. No manual step needed.

### Step 6: Launch the Application

```powershell
python main.py
```

---

## 📖 Example Workflow

1. **Launch** the app with `python main.py`.
2. Click **📂 Upload Book** in the sidebar.
3. Select a `.txt` file (try `data/books/test_chapters.txt`).
4. Click **🔍 Analyze Book** — analysis runs in the background.
5. **Overview** tab shows summary cards and story signature.
6. **Readability** tab shows Flesch scores and word statistics.
7. **Topics** tab shows discovered word clusters (LDA).
8. **Sentiment** tab shows the emotional arc chart and interpretation.
9. **History** tab shows previously analyzed books (requires MySQL).
10. Use **Export JSON** or **Export CSV** to save reports.

---

## 📊 Features

### Readability Analysis
- Word count, sentence count, unique words
- Vocabulary richness (type-token ratio)
- Flesch Reading Ease score
- Flesch-Kincaid Grade Level
- Average words per sentence and word length

### Chapter-Level Sentiment Analysis
- VADER compound, positive, neutral, negative scores per chapter
- Emotional arc line chart with score annotations
- Stacked bar chart showing sentiment breakdown
- Automated interpretation of sentiment trends

### LDA Topic Modeling
- Configurable number of topics (default: 5, auto-adjusted)
- Top keywords with probability weights per topic
- Topic distribution visualization
- Handles insufficient text gracefully

### Story Signature
- Rule-based classification using sentiment indicators
- Labels: positive/negative/mixed tendency, emotionally stable/dynamic, arc direction
- Clear disclaimer about interpretive limitations

### Database & History
- Automatic MySQL database and table creation
- Parameterized SQL queries (no injection risk)
- Browse and inspect past analyses
- Delete old entries
- Graceful offline mode when MySQL is unavailable

### Report Export
- JSON export with full analysis data
- CSV export with summary and chapter-level data

---

## ⚠️ Known Limitations

1. **Chapter detection** relies on common heading patterns. Unconventional formats may not be detected.
2. **VADER sentiment** measures word-level valence. It does not understand irony, sarcasm, or literary devices.
3. **LDA topics** are statistical word clusters. They require human interpretation and may not correspond to meaningful themes.
4. **Story signature** is an exploratory heuristic, not a validated literary classification system.
5. **Only `.txt` files** are supported. PDF, EPUB, and DOCX are not handled.
6. **Large books** (500K+ words) may take 30+ seconds for full analysis.
7. **Readability formulas** are designed for English text and may give unreliable results for other languages.

---

## 🔮 Future Enhancements

- PDF and EPUB file support
- Named Entity Recognition (NER) for character analysis
- Word cloud visualization
- Comparative analysis across multiple books
- Interactive topic exploration
- Export to PDF reports
- Support for non-English languages
- Cloud database option

---

## 🧪 Running Tests

```powershell
python -m pytest tests/ -v
```

---

## 🎓 College Presentation Summary

> **E-Reader Text Analytics Engine** is a desktop Data Science application that demonstrates practical NLP on literary texts. It combines text preprocessing (NLTK), readability scoring (textstat), unsupervised topic discovery (Gensim LDA), and sentiment analysis (VADER) into an interactive Tkinter dashboard. The application processes `.txt` books, extracts chapters automatically, calculates readability metrics, discovers latent topics, visualizes emotional arcs, and persists results in MySQL. It showcases a complete data pipeline — from raw text ingestion through analysis to visualization and storage — using industry-standard Python libraries. The project handles edge cases gracefully, provides honest disclaimers about analytical limitations, and follows clean code practices with modular architecture, type hints, and comprehensive testing.

---

## 📝 License

Academic project — for educational use.
