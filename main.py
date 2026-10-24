"""
E-Reader Text Analytics Engine

A desktop application for NLP-based analysis of text books.
Launch this file to start the application.
"""

import sys
import os
import tkinter as tk

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui import EReaderApp


def main():
    """Initialize and run the E-Reader application."""
    root = tk.Tk()
    app = EReaderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()