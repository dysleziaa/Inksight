# InkSight 📚

**See beyond the words.**

InkSight is a Python-based book text analytics project that uses Natural Language Processing (NLP) to explore the structure, readability, topics, and emotional tone of literary texts.

## Features

- Text preprocessing and chapter extraction
- Readability analysis using `textstat`
- Sentiment analysis using VADER
- Topic modeling using LDA
- Chapter-level sentiment and emotional arc analysis
- Visualizations with Matplotlib
- Reading history stored in MySQL
- JSON and CSV export
- Desktop interface built with Tkinter

## Tech Stack

Python · NLTK · Gensim · VADER · Textstat · Matplotlib · Tkinter · MySQL

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the NLTK resources required by the application.
5. Configure MySQL and create the `ereader_analytics` database.
6. Configure database credentials using environment variables or your local `.env` file.

## Run

```bash
python main.py
```

## Tests

```bash
pytest
```
