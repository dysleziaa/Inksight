"""
E-Reader Text Analytics Engine - Tkinter Desktop UI

A literary analytics dashboard built with Tkinter and ttk.
Features a navigation sidebar, book upload, analysis display,
emotional arc charts, and reading history.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Optional
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.text_processor import TextProcessor
from src.chapter_extractor import extract_chapters
from src.readability_analyzer import ReadabilityAnalyzer
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_modeler import TopicModeler
from src.visualization import (
    create_emotional_arc_figure,
    create_topic_distribution_figure,
    create_sentiment_breakdown_figure,
    COLORS,
)
from src.database import DatabaseManager
from src.report_exporter import ReportExporter


# ── Theme Colors ──────────────────────────────────────────────────────
BG_DARK = '#1A1A2E'
BG_SIDEBAR = '#16213E'
BG_CARD = '#1F2B47'
BG_CARD_HOVER = '#263352'
BG_INPUT = '#2A3A5C'
FG_PRIMARY = '#E2E8F0'
FG_SECONDARY = '#94A3B8'
FG_ACCENT = '#818CF8'
FG_ACCENT_HOVER = '#A5B4FC'
FG_SUCCESS = '#34D399'
FG_WARNING = '#FBBF24'
FG_ERROR = '#F87171'
BORDER_COLOR = '#334155'
BUTTON_BG = '#4F46E5'
BUTTON_BG_HOVER = '#6366F1'
BUTTON_BG_DISABLED = '#374151'
BUTTON_FG = '#FFFFFF'


class EReaderApp:
    """Main application class for the E-Reader Text Analytics Engine."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("E-Reader Text Analytics Engine")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 650)
        self.root.configure(bg=BG_DARK)

        # Set window icon (suppress errors if not found)
        try:
            self.root.iconbitmap(default='')
        except Exception:
            pass

        # ── Module instances ──
        self.text_processor = TextProcessor()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        self.db = DatabaseManager()
        self.exporter = ReportExporter()

        # ── State ──
        self._current_file = None
        self._analysis_results = None
        self._is_analyzing = False
        self._chart_canvases = []  # Track embedded matplotlib canvases

        # ── Try database connection ──
        self._db_connected = self.db.connect()

        # ── Build UI ──
        self._configure_styles()
        self._build_layout()
        self._update_db_status()

    # ══════════════════════════════════════════════════════════════════
    # STYLES
    # ══════════════════════════════════════════════════════════════════
    def _configure_styles(self):
        """Configure ttk styles for the dark theme."""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # General
        self.style.configure('.', background=BG_DARK, foreground=FG_PRIMARY,
                             font=('Segoe UI', 10))

        # Frame
        self.style.configure('Card.TFrame', background=BG_CARD, relief='flat')
        self.style.configure('Sidebar.TFrame', background=BG_SIDEBAR)
        self.style.configure('Dark.TFrame', background=BG_DARK)

        # Labels
        self.style.configure('Title.TLabel', background=BG_DARK, foreground=FG_PRIMARY,
                             font=('Segoe UI', 20, 'bold'))
        self.style.configure('Subtitle.TLabel', background=BG_DARK, foreground=FG_SECONDARY,
                             font=('Segoe UI', 11))
        self.style.configure('Card.TLabel', background=BG_CARD, foreground=FG_PRIMARY,
                             font=('Segoe UI', 10))
        self.style.configure('CardTitle.TLabel', background=BG_CARD, foreground=FG_ACCENT,
                             font=('Segoe UI', 12, 'bold'))
        self.style.configure('CardValue.TLabel', background=BG_CARD, foreground=FG_PRIMARY,
                             font=('Segoe UI', 22, 'bold'))
        self.style.configure('CardUnit.TLabel', background=BG_CARD, foreground=FG_SECONDARY,
                             font=('Segoe UI', 9))
        self.style.configure('Sidebar.TLabel', background=BG_SIDEBAR, foreground=FG_PRIMARY,
                             font=('Segoe UI', 10))
        self.style.configure('SidebarTitle.TLabel', background=BG_SIDEBAR,
                             foreground=FG_ACCENT, font=('Segoe UI', 14, 'bold'))
        self.style.configure('Status.TLabel', background=BG_DARK, foreground=FG_SECONDARY,
                             font=('Segoe UI', 9))
        self.style.configure('Success.TLabel', background=BG_DARK, foreground=FG_SUCCESS,
                             font=('Segoe UI', 9))
        self.style.configure('Warning.TLabel', background=BG_DARK, foreground=FG_WARNING,
                             font=('Segoe UI', 9))
        self.style.configure('Error.TLabel', background=BG_DARK, foreground=FG_ERROR,
                             font=('Segoe UI', 9))

        # Buttons
        self.style.configure('Accent.TButton', background=BUTTON_BG, foreground=BUTTON_FG,
                             font=('Segoe UI', 10, 'bold'), padding=(15, 8))
        self.style.map('Accent.TButton',
                       background=[('active', BUTTON_BG_HOVER), ('disabled', BUTTON_BG_DISABLED)],
                       foreground=[('disabled', FG_SECONDARY)])

        self.style.configure('Sidebar.TButton', background=BG_SIDEBAR, foreground=FG_PRIMARY,
                             font=('Segoe UI', 10), padding=(10, 8), anchor='w')
        self.style.map('Sidebar.TButton',
                       background=[('active', BG_CARD)])

        self.style.configure('Small.TButton', background=BG_INPUT, foreground=FG_PRIMARY,
                             font=('Segoe UI', 9), padding=(8, 4))
        self.style.map('Small.TButton',
                       background=[('active', BG_CARD_HOVER)])

        # Progressbar
        self.style.configure('Accent.Horizontal.TProgressbar',
                             background=FG_ACCENT, troughcolor=BG_INPUT)

        # Separator
        self.style.configure('Dark.TSeparator', background=BORDER_COLOR)

        # Notebook (tabs)
        self.style.configure('TNotebook', background=BG_DARK, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=BG_CARD, foreground=FG_SECONDARY,
                             font=('Segoe UI', 10), padding=(15, 8))
        self.style.map('TNotebook.Tab',
                       background=[('selected', BG_DARK)],
                       foreground=[('selected', FG_ACCENT)])

        # Treeview
        self.style.configure('Treeview', background=BG_CARD, foreground=FG_PRIMARY,
                             fieldbackground=BG_CARD, font=('Segoe UI', 9),
                             rowheight=28)
        self.style.configure('Treeview.Heading', background=BG_INPUT, foreground=FG_ACCENT,
                             font=('Segoe UI', 9, 'bold'))
        self.style.map('Treeview', background=[('selected', BG_INPUT)],
                       foreground=[('selected', FG_PRIMARY)])

    # ══════════════════════════════════════════════════════════════════
    # LAYOUT
    # ══════════════════════════════════════════════════════════════════
    def _build_layout(self):
        """Build the main application layout."""
        # ── Main container ──
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──
        self._build_sidebar()

        # ── Content area ──
        self.content_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.content_frame.grid(row=0, column=1, sticky='nsew')
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        # ── Header ──
        self._build_header()

        # ── Notebook (tab view) ──
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0, 5))

        self._build_overview_tab()
        self._build_readability_tab()
        self._build_topics_tab()
        self._build_sentiment_tab()
        self._build_history_tab()

        # ── Status bar ──
        self._build_status_bar()

    def _build_sidebar(self):
        """Build the navigation sidebar."""
        sidebar = ttk.Frame(self.root, style='Sidebar.TFrame', width=220)
        sidebar.grid(row=0, column=0, sticky='ns')
        sidebar.grid_propagate(False)

        # App title
        ttk.Label(sidebar, text="📖 E-Reader", style='SidebarTitle.TLabel').pack(
            pady=(20, 2), padx=15, anchor='w')
        ttk.Label(sidebar, text="Text Analytics Engine",
                  style='Sidebar.TLabel').pack(padx=15, anchor='w')

        ttk.Separator(sidebar, orient='horizontal',
                      style='Dark.TSeparator').pack(fill='x', padx=15, pady=15)

        # Upload button
        self.btn_upload = ttk.Button(sidebar, text="📂  Upload Book",
                                      style='Accent.TButton', command=self._on_upload)
        self.btn_upload.pack(padx=15, fill='x', pady=(0, 8))

        # Analyze button
        self.btn_analyze = ttk.Button(sidebar, text="🔍  Analyze Book",
                                       style='Accent.TButton', command=self._on_analyze)
        self.btn_analyze.pack(padx=15, fill='x', pady=(0, 8))
        self.btn_analyze.state(['disabled'])

        ttk.Separator(sidebar, orient='horizontal',
                      style='Dark.TSeparator').pack(fill='x', padx=15, pady=10)

        # Navigation buttons
        nav_items = [
            ("📊  Overview", lambda: self.notebook.select(0)),
            ("📖  Readability", lambda: self.notebook.select(1)),
            ("🏷️  Topics", lambda: self.notebook.select(2)),
            ("💭  Sentiment", lambda: self.notebook.select(3)),
            ("📚  History", lambda: self._show_history_tab()),
        ]
        for text, cmd in nav_items:
            ttk.Button(sidebar, text=text, style='Sidebar.TButton',
                       command=cmd).pack(padx=10, fill='x', pady=1)

        ttk.Separator(sidebar, orient='horizontal',
                      style='Dark.TSeparator').pack(fill='x', padx=15, pady=10)

        # Export buttons
        ttk.Label(sidebar, text="Export Report", style='Sidebar.TLabel').pack(
            padx=15, anchor='w', pady=(0, 5))
        self.btn_export_json = ttk.Button(sidebar, text="💾  Export JSON",
                                           style='Small.TButton',
                                           command=lambda: self._on_export('json'))
        self.btn_export_json.pack(padx=15, fill='x', pady=2)
        self.btn_export_json.state(['disabled'])

        self.btn_export_csv = ttk.Button(sidebar, text="📄  Export CSV",
                                          style='Small.TButton',
                                          command=lambda: self._on_export('csv'))
        self.btn_export_csv.pack(padx=15, fill='x', pady=2)
        self.btn_export_csv.state(['disabled'])

        # DB status at bottom
        spacer = ttk.Frame(sidebar, style='Sidebar.TFrame')
        spacer.pack(fill='both', expand=True)

        self.db_status_label = ttk.Label(sidebar, text="", style='Sidebar.TLabel')
        self.db_status_label.pack(padx=15, pady=(0, 15), anchor='w')

    def _build_header(self):
        """Build the header area with book info."""
        header = ttk.Frame(self.content_frame, style='Dark.TFrame')
        header.grid(row=0, column=0, sticky='ew', padx=10, pady=(10, 5))

        self.lbl_book_title = ttk.Label(header, text="No book loaded",
                                         style='Title.TLabel')
        self.lbl_book_title.pack(anchor='w')

        self.lbl_book_info = ttk.Label(header, text="Upload a .txt file to begin analysis",
                                        style='Subtitle.TLabel')
        self.lbl_book_info.pack(anchor='w', pady=(2, 0))

    def _build_status_bar(self):
        """Build the bottom status bar."""
        status_frame = ttk.Frame(self.content_frame, style='Dark.TFrame')
        status_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        status_frame.grid_columnconfigure(0, weight=1)

        self.lbl_status = ttk.Label(status_frame, text="Ready", style='Status.TLabel')
        self.lbl_status.grid(row=0, column=0, sticky='w')

        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate',
                                             style='Accent.Horizontal.TProgressbar',
                                             length=200)
        self.progress_bar.grid(row=0, column=1, sticky='e', padx=(10, 0))

    # ══════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════
    def _build_overview_tab(self):
        """Build the Overview tab with summary cards."""
        frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(frame, text='  📊 Overview  ')
        frame.grid_columnconfigure(0, weight=1)

        # Scrollable container
        canvas = tk.Canvas(frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        self.overview_inner = ttk.Frame(canvas, style='Dark.TFrame')

        self.overview_inner.bind('<Configure>',
                                 lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas_window = canvas.create_window((0, 0), window=self.overview_inner, anchor='nw')

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel, add='+')

        # Placeholder
        self.overview_placeholder = ttk.Label(
            self.overview_inner, text="\n\n📖  Upload and analyze a book to see results here.\n\n",
            style='Subtitle.TLabel', anchor='center')
        self.overview_placeholder.pack(expand=True, fill='x', pady=50)

    def _build_readability_tab(self):
        """Build the Readability tab."""
        frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(frame, text='  📖 Readability  ')

        canvas = tk.Canvas(frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        self.readability_inner = ttk.Frame(canvas, style='Dark.TFrame')
        self.readability_inner.bind('<Configure>',
                                    lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        cw = canvas.create_window((0, 0), window=self.readability_inner, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.readability_placeholder = ttk.Label(
            self.readability_inner,
            text="\n\n📖  Readability metrics will appear here after analysis.\n\n",
            style='Subtitle.TLabel', anchor='center')
        self.readability_placeholder.pack(expand=True, fill='x', pady=50)

    def _build_topics_tab(self):
        """Build the Topics tab."""
        frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(frame, text='  🏷️ Topics  ')

        canvas = tk.Canvas(frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        self.topics_inner = ttk.Frame(canvas, style='Dark.TFrame')
        self.topics_inner.bind('<Configure>',
                                lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        cw = canvas.create_window((0, 0), window=self.topics_inner, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.topics_placeholder = ttk.Label(
            self.topics_inner,
            text="\n\n🏷️  Topic modeling results will appear here after analysis.\n\n",
            style='Subtitle.TLabel', anchor='center')
        self.topics_placeholder.pack(expand=True, fill='x', pady=50)

    def _build_sentiment_tab(self):
        """Build the Sentiment / Emotional Arc tab."""
        frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(frame, text='  💭 Sentiment  ')

        canvas = tk.Canvas(frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        self.sentiment_inner = ttk.Frame(canvas, style='Dark.TFrame')
        self.sentiment_inner.bind('<Configure>',
                                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        cw = canvas.create_window((0, 0), window=self.sentiment_inner, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.sentiment_placeholder = ttk.Label(
            self.sentiment_inner,
            text="\n\n💭  Sentiment analysis will appear here after analysis.\n\n",
            style='Subtitle.TLabel', anchor='center')
        self.sentiment_placeholder.pack(expand=True, fill='x', pady=50)

    def _build_history_tab(self):
        """Build the Reading History tab."""
        frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(frame, text='  📚 History  ')
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ttk.Frame(frame, style='Dark.TFrame')
        header.pack(fill='x', padx=10, pady=10)
        ttk.Label(header, text="📚 Reading History", style='CardTitle.TLabel',
                  background=BG_DARK).pack(side='left')
        ttk.Button(header, text="🔄 Refresh", style='Small.TButton',
                   command=self._load_history).pack(side='right', padx=(5, 0))
        ttk.Button(header, text="🔍 View Details", style='Small.TButton',
                   command=self._view_history_details).pack(side='right', padx=(5, 0))
        ttk.Button(header, text="🗑️ Delete", style='Small.TButton',
                   command=self._delete_history_entry).pack(side='right')

        # History treeview
        tree_frame = ttk.Frame(frame, style='Dark.TFrame')
        tree_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        columns = ('id', 'title', 'date', 'words', 'readability', 'chapters')
        self.history_tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                          selectmode='browse')
        self.history_tree.heading('id', text='ID')
        self.history_tree.heading('title', text='Title')
        self.history_tree.heading('date', text='Analyzed')
        self.history_tree.heading('words', text='Words')
        self.history_tree.heading('readability', text='Readability')
        self.history_tree.heading('chapters', text='Chapters')

        self.history_tree.column('id', width=40, anchor='center')
        self.history_tree.column('title', width=250)
        self.history_tree.column('date', width=150, anchor='center')
        self.history_tree.column('words', width=80, anchor='center')
        self.history_tree.column('readability', width=100, anchor='center')
        self.history_tree.column('chapters', width=70, anchor='center')

        tree_scroll = ttk.Scrollbar(tree_frame, orient='vertical',
                                     command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=tree_scroll.set)
        self.history_tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')

        # Detail panel
        self.history_detail = ttk.Frame(frame, style='Dark.TFrame')
        self.history_detail.pack(fill='x', padx=10, pady=(0, 10))

        self.history_detail_text = tk.Text(
            self.history_detail, bg=BG_CARD, fg=FG_PRIMARY,
            font=('Consolas', 10), height=8, wrap='word',
            insertbackground=FG_PRIMARY, relief='flat', padx=10, pady=10)
        self.history_detail_text.pack(fill='x')
        self.history_detail_text.insert('1.0', 'Select a book and click "View Details" to see its analysis.')
        self.history_detail_text.config(state='disabled')

    # ══════════════════════════════════════════════════════════════════
    # UI HELPERS
    # ══════════════════════════════════════════════════════════════════
    def _create_metric_card(self, parent: ttk.Frame, title: str, value: str,
                             unit: str = '', row: int = 0, col: int = 0) -> ttk.Frame:
        """Create a styled metric card widget."""
        card = ttk.Frame(parent, style='Card.TFrame', padding=15)
        card.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)

        ttk.Label(card, text=title, style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(card, text=str(value), style='CardValue.TLabel').pack(anchor='w', pady=(5, 0))
        if unit:
            ttk.Label(card, text=unit, style='CardUnit.TLabel').pack(anchor='w')

        return card

    def _create_info_card(self, parent: ttk.Frame, title: str, text: str,
                           fg_color: str = FG_PRIMARY) -> ttk.Frame:
        """Create an info card with wrapped text."""
        card = ttk.Frame(parent, style='Card.TFrame', padding=15)
        card.pack(fill='x', padx=5, pady=5)

        ttk.Label(card, text=title, style='CardTitle.TLabel').pack(anchor='w')
        lbl = tk.Label(card, text=text, bg=BG_CARD, fg=fg_color,
                       font=('Segoe UI', 10), wraplength=700, justify='left')
        lbl.pack(anchor='w', pady=(5, 0))

        return card

    def _clear_frame(self, frame: ttk.Frame):
        """Remove all children from a frame."""
        for widget in frame.winfo_children():
            widget.destroy()

    def _destroy_chart_canvases(self):
        """Properly destroy matplotlib canvases."""
        for canvas in self._chart_canvases:
            try:
                canvas.get_tk_widget().destroy()
            except Exception:
                pass
        self._chart_canvases.clear()

    def _embed_figure(self, parent: ttk.Frame, figure: Figure) -> FigureCanvasTkAgg:
        """Embed a matplotlib figure into a tkinter frame."""
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        self._chart_canvases.append(canvas)
        return canvas

    def _set_status(self, text: str, style: str = 'Status.TLabel'):
        """Update the status bar text."""
        self.lbl_status.configure(text=text, style=style)

    def _update_db_status(self):
        """Update the database status indicator."""
        if self._db_connected:
            self.db_status_label.configure(text="🟢 MySQL Connected",
                                            foreground=FG_SUCCESS)
        else:
            msg = "🔴 MySQL Offline"
            self.db_status_label.configure(text=msg, foreground=FG_ERROR)

    # ══════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════
    def _on_upload(self):
        """Handle the Upload Book button click."""
        file_path = filedialog.askopenfilename(
            title="Select a Book (.txt)",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'books')
        )
        if not file_path:
            return

        try:
            # Validate by loading
            raw_text = self.text_processor.load_text(file_path)
            self._current_file = file_path
            title = self.text_processor.detect_title(file_path, raw_text)
            stats = self.text_processor.get_basic_stats(
                self.text_processor.normalize_whitespace(raw_text))

            self.lbl_book_title.configure(text=f"📖 {title}")
            self.lbl_book_info.configure(
                text=f"File: {os.path.basename(file_path)}  |  "
                     f"{stats['character_count']:,} characters  |  "
                     f"{stats['word_count']:,} words  |  "
                     f"{stats['sentence_count']:,} sentences")

            self.btn_analyze.state(['!disabled'])
            self._set_status(f"Book loaded: {os.path.basename(file_path)}", 'Success.TLabel')

        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("File Error", str(e))
            self._set_status(f"Error loading file", 'Error.TLabel')
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error loading file:\n{str(e)}")
            self._set_status("Error loading file", 'Error.TLabel')

    def _on_analyze(self):
        """Handle the Analyze Book button click. Runs analysis in a background thread."""
        if self._is_analyzing or not self._current_file:
            return

        self._is_analyzing = True
        self.btn_analyze.state(['disabled'])
        self.btn_upload.state(['disabled'])
        self.btn_export_json.state(['disabled'])
        self.btn_export_csv.state(['disabled'])
        self.progress_bar.start(15)
        self._set_status("Analyzing book... This may take a moment.", 'Warning.TLabel')

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        """Execute the full analysis pipeline in a background thread."""
        try:
            file_path = self._current_file

            # Step 1: Process text
            self._update_status_safe("Processing text...")
            processed = self.text_processor.process(file_path)

            # Step 2: Extract chapters
            self._update_status_safe("Extracting chapters...")
            chapters = extract_chapters(processed['normalized_text'])

            # Step 3: Readability
            self._update_status_safe("Calculating readability metrics...")
            readability = self.readability_analyzer.analyze(processed['normalized_text'])

            # Step 4: Sentiment analysis
            self._update_status_safe("Analyzing sentiment...")
            chapter_sentiments = self.sentiment_analyzer.analyze_chapters(chapters)
            interpretation = self.sentiment_analyzer.generate_interpretation(chapter_sentiments)
            story_signature = self.sentiment_analyzer.generate_story_signature(chapter_sentiments)

            # Step 5: Topic modeling
            self._update_status_safe("Building topic model...")
            chapter_tokens = [
                self.text_processor.get_tokens_for_topic_modeling(ch['text'])
                for ch in chapters
            ]
            topic_results = self.topic_modeler.build_model(chapter_tokens)

            # Store results
            self._analysis_results = {
                'processed': processed,
                'chapters': chapters,
                'readability': readability,
                'chapter_sentiments': chapter_sentiments,
                'interpretation': interpretation,
                'story_signature': story_signature,
                'topic_results': topic_results,
            }

            # Step 6: Save to database
            if self._db_connected:
                self._update_status_safe("Saving to database...")
                success, msg = self.db.save_analysis(
                    processed, chapters, chapter_sentiments,
                    topic_results, readability
                )
                if not success:
                    self._show_warning_safe("Database Warning",
                                            f"Analysis completed but could not be saved to database:\n{msg}")

            # Update UI on main thread
            self.root.after(0, self._display_results)

        except Exception as e:
            self.root.after(0, lambda: self._on_analysis_error(str(e)))

    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error on the main thread."""
        self._is_analyzing = False
        self.progress_bar.stop()
        self.btn_analyze.state(['!disabled'])
        self.btn_upload.state(['!disabled'])
        self._set_status(f"Analysis failed: {error_msg}", 'Error.TLabel')
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_msg}")

    def _update_status_safe(self, text: str):
        """Thread-safe status update."""
        self.root.after(0, lambda: self._set_status(text, 'Warning.TLabel'))

    def _show_warning_safe(self, title: str, message: str):
        """Thread-safe warning dialog."""
        self.root.after(0, lambda: messagebox.showwarning(title, message))

    # ══════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════════════════════════════
    def _display_results(self):
        """Populate all tabs with analysis results. Must run on main thread."""
        self._is_analyzing = False
        self.progress_bar.stop()
        self.btn_analyze.state(['!disabled'])
        self.btn_upload.state(['!disabled'])
        self.btn_export_json.state(['!disabled'])
        self.btn_export_csv.state(['!disabled'])

        results = self._analysis_results
        if not results:
            return

        self._destroy_chart_canvases()
        self._display_overview(results)
        self._display_readability(results)
        self._display_topics(results)
        self._display_sentiment(results)

        # Switch to overview tab
        self.notebook.select(0)

        # Status message
        db_msg = ""
        if self._db_connected:
            db_msg = " | Results saved to database"
        else:
            db_msg = " | ⚠ Database offline — results not saved"

        self._set_status(f"Analysis complete{db_msg}", 'Success.TLabel')

    def _display_overview(self, results: Dict):
        """Populate the Overview tab."""
        self._clear_frame(self.overview_inner)

        readability = results['readability']
        chapters = results['chapters']
        processed = results['processed']
        signature = results['story_signature']

        # ── Summary Cards Grid ──
        cards_frame = ttk.Frame(self.overview_inner, style='Dark.TFrame')
        cards_frame.pack(fill='x', padx=5, pady=5)
        for i in range(4):
            cards_frame.grid_columnconfigure(i, weight=1)

        self._create_metric_card(cards_frame, "Total Words",
                                  f"{readability['word_count']:,}", "words", 0, 0)
        self._create_metric_card(cards_frame, "Sentences",
                                  f"{readability['sentence_count']:,}", "sentences", 0, 1)
        self._create_metric_card(cards_frame, "Chapters",
                                  str(len(chapters)),
                                  "detected" if len(chapters) > 1 else "section", 0, 2)
        self._create_metric_card(cards_frame, "Unique Words",
                                  f"{readability['unique_word_count']:,}",
                                  f"richness: {readability['vocabulary_richness']:.2%}", 0, 3)

        # Row 2
        fre = readability.get('flesch_reading_ease')
        fre_display = f"{fre:.1f}" if fre is not None else "N/A"
        fkg = readability.get('flesch_kincaid_grade')
        fkg_display = f"{fkg:.1f}" if fkg is not None else "N/A"
        level = self.readability_analyzer.get_reading_level_description(fre)

        self._create_metric_card(cards_frame, "Flesch Reading Ease",
                                  fre_display, level, 1, 0)
        self._create_metric_card(cards_frame, "Grade Level",
                                  fkg_display, "Flesch-Kincaid", 1, 1)
        self._create_metric_card(cards_frame, "Avg Words/Sentence",
                                  f"{readability['avg_words_per_sentence']:.1f}",
                                  "words per sentence", 1, 2)
        self._create_metric_card(cards_frame, "Avg Word Length",
                                  f"{readability['avg_word_length']:.1f}",
                                  "characters", 1, 3)

        # ── Story Signature ──
        sig_card = ttk.Frame(self.overview_inner, style='Card.TFrame', padding=15)
        sig_card.pack(fill='x', padx=10, pady=10)

        ttk.Label(sig_card, text="🎭 Story Signature", style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(sig_card, text=signature.get('label', 'N/A'),
                  style='CardValue.TLabel',
                  font=('Segoe UI', 16, 'bold')).pack(anchor='w', pady=(5, 0))

        # Components
        components = signature.get('components', {})
        if components:
            comp_text = (f"Avg Sentiment: {components.get('avg_sentiment', 0):.3f}  |  "
                         f"Variability: {components.get('sentiment_variability', 0):.3f}  |  "
                         f"Arc Direction: {components.get('arc_direction', 0):.3f}  |  "
                         f"Final: {components.get('final_sentiment', 0):.3f}")
            tk.Label(sig_card, text=comp_text, bg=BG_CARD, fg=FG_SECONDARY,
                     font=('Segoe UI', 9)).pack(anchor='w', pady=(5, 0))

        # Note
        note = signature.get('note', '')
        if note:
            tk.Label(sig_card, text=note, bg=BG_CARD, fg=FG_SECONDARY,
                     font=('Segoe UI', 9, 'italic'), wraplength=800,
                     justify='left').pack(anchor='w', pady=(8, 0))

        # ── Chapter detection note ──
        first_ch = chapters[0] if chapters else {}
        if first_ch.get('detection_note'):
            self._create_info_card(self.overview_inner, "ℹ️ Chapter Detection",
                                    first_ch['detection_note'], FG_WARNING)

    def _display_readability(self, results: Dict):
        """Populate the Readability tab."""
        self._clear_frame(self.readability_inner)
        readability = results['readability']

        ttk.Label(self.readability_inner, text="📖 Readability Analysis",
                  style='Title.TLabel').pack(anchor='w', padx=10, pady=(10, 5))

        # Metrics grid
        grid = ttk.Frame(self.readability_inner, style='Dark.TFrame')
        grid.pack(fill='x', padx=5, pady=5)
        for i in range(3):
            grid.grid_columnconfigure(i, weight=1)

        metrics = [
            ("Word Count", f"{readability['word_count']:,}", "total words"),
            ("Sentence Count", f"{readability['sentence_count']:,}", "total sentences"),
            ("Unique Words", f"{readability['unique_word_count']:,}", "distinct words"),
            ("Avg Words/Sentence", f"{readability['avg_words_per_sentence']:.2f}", ""),
            ("Avg Word Length", f"{readability['avg_word_length']:.2f}", "characters"),
            ("Vocabulary Richness", f"{readability['vocabulary_richness']:.4f}",
             "unique/total ratio"),
        ]
        for i, (title, value, unit) in enumerate(metrics):
            self._create_metric_card(grid, title, value, unit, i // 3, i % 3)

        # Flesch scores
        fre = readability.get('flesch_reading_ease')
        fkg = readability.get('flesch_kincaid_grade')

        flesch_frame = ttk.Frame(self.readability_inner, style='Dark.TFrame')
        flesch_frame.pack(fill='x', padx=5, pady=5)
        flesch_frame.grid_columnconfigure(0, weight=1)
        flesch_frame.grid_columnconfigure(1, weight=1)

        if fre is not None:
            level = self.readability_analyzer.get_reading_level_description(fre)
            self._create_metric_card(flesch_frame, "Flesch Reading Ease",
                                      f"{fre:.2f}", level, 0, 0)
        if fkg is not None:
            self._create_metric_card(flesch_frame, "Flesch-Kincaid Grade Level",
                                      f"{fkg:.2f}", "grade level", 0, 1)

        # Error message
        if readability.get('error'):
            self._create_info_card(self.readability_inner, "⚠️ Note",
                                    readability['error'], FG_WARNING)

    def _display_topics(self, results: Dict):
        """Populate the Topics tab."""
        self._clear_frame(self.topics_inner)
        topic_results = results['topic_results']

        ttk.Label(self.topics_inner, text="🏷️ Topic Modeling (LDA)",
                  style='Title.TLabel').pack(anchor='w', padx=10, pady=(10, 5))

        if not topic_results.get('success'):
            self._create_info_card(self.topics_inner, "⚠️ Topic Modeling",
                                    topic_results.get('error', 'Unknown error'), FG_WARNING)
            return

        # Note about interpretation
        self._create_info_card(self.topics_inner, "ℹ️ About These Topics",
                                topic_results.get('note', ''), FG_SECONDARY)

        # Topic cards
        for topic in topic_results.get('topics', []):
            card = ttk.Frame(self.topics_inner, style='Card.TFrame', padding=15)
            card.pack(fill='x', padx=10, pady=5)

            ttk.Label(card, text=f"Topic {topic['topic_number']}",
                      style='CardTitle.TLabel').pack(anchor='w')

            # Keywords with weights
            kw_frame = ttk.Frame(card, style='Card.TFrame')
            kw_frame.pack(fill='x', pady=(5, 0))

            for word, weight in topic.get('keyword_weights', []):
                tag = tk.Label(kw_frame, text=f" {word} ({weight:.3f}) ",
                               bg=BG_INPUT, fg=FG_ACCENT, font=('Segoe UI', 9),
                               padx=6, pady=2)
                tag.pack(side='left', padx=2, pady=2)

        # Topic distribution chart
        fig = create_topic_distribution_figure(topic_results, figsize=(9, 4))
        if fig:
            chart_frame = ttk.Frame(self.topics_inner, style='Card.TFrame', padding=10)
            chart_frame.pack(fill='x', padx=10, pady=10)
            ttk.Label(chart_frame, text="Topic Distribution",
                      style='CardTitle.TLabel').pack(anchor='w', pady=(0, 5))
            self._embed_figure(chart_frame, fig)

    def _display_sentiment(self, results: Dict):
        """Populate the Sentiment tab."""
        self._clear_frame(self.sentiment_inner)
        chapter_sentiments = results['chapter_sentiments']
        interpretation = results['interpretation']

        ttk.Label(self.sentiment_inner, text="💭 Sentiment Analysis & Emotional Arc",
                  style='Title.TLabel').pack(anchor='w', padx=10, pady=(10, 5))

        # ── Emotional Arc Chart ──
        arc_frame = ttk.Frame(self.sentiment_inner, style='Card.TFrame', padding=10)
        arc_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(arc_frame, text="Emotional Arc",
                  style='CardTitle.TLabel').pack(anchor='w', pady=(0, 5))

        fig = create_emotional_arc_figure(chapter_sentiments, figsize=(9, 4))
        self._embed_figure(arc_frame, fig)

        # ── Interpretation ──
        interp_card = ttk.Frame(self.sentiment_inner, style='Card.TFrame', padding=15)
        interp_card.pack(fill='x', padx=10, pady=5)
        ttk.Label(interp_card, text="📝 Interpretation",
                  style='CardTitle.TLabel').pack(anchor='w')

        for key in ['overall', 'variation', 'arc']:
            text = interpretation.get(key, '')
            if text:
                tk.Label(interp_card, text=f"• {text}", bg=BG_CARD, fg=FG_PRIMARY,
                         font=('Segoe UI', 10), wraplength=750, justify='left',
                         anchor='w').pack(anchor='w', pady=2)

        # Disclaimer
        disclaimer = interpretation.get('disclaimer', '')
        if disclaimer:
            tk.Label(interp_card, text=disclaimer, bg=BG_CARD, fg=FG_SECONDARY,
                     font=('Segoe UI', 9, 'italic'), wraplength=750,
                     justify='left').pack(anchor='w', pady=(8, 0))

        # ── Chapter Sentiment Table ──
        table_card = ttk.Frame(self.sentiment_inner, style='Card.TFrame', padding=15)
        table_card.pack(fill='x', padx=10, pady=5)
        ttk.Label(table_card, text="Chapter-Level Sentiment Scores",
                  style='CardTitle.TLabel').pack(anchor='w', pady=(0, 5))

        cols = ('chapter', 'compound', 'positive', 'neutral', 'negative')
        tree = ttk.Treeview(table_card, columns=cols, show='headings', height=min(len(chapter_sentiments), 15))
        tree.heading('chapter', text='Chapter')
        tree.heading('compound', text='Compound')
        tree.heading('positive', text='Positive')
        tree.heading('neutral', text='Neutral')
        tree.heading('negative', text='Negative')

        tree.column('chapter', width=200)
        tree.column('compound', width=100, anchor='center')
        tree.column('positive', width=100, anchor='center')
        tree.column('neutral', width=100, anchor='center')
        tree.column('negative', width=100, anchor='center')

        for cs in chapter_sentiments:
            s = cs['sentiment']
            tree.insert('', 'end', values=(
                cs['title'],
                f"{s['compound']:.4f}",
                f"{s['pos']:.4f}",
                f"{s['neu']:.4f}",
                f"{s['neg']:.4f}",
            ))
        tree.pack(fill='x')

        # ── Sentiment Breakdown Chart ──
        breakdown_frame = ttk.Frame(self.sentiment_inner, style='Card.TFrame', padding=10)
        breakdown_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(breakdown_frame, text="Sentiment Breakdown",
                  style='CardTitle.TLabel').pack(anchor='w', pady=(0, 5))

        fig2 = create_sentiment_breakdown_figure(chapter_sentiments, figsize=(9, 4))
        self._embed_figure(breakdown_frame, fig2)

    # ══════════════════════════════════════════════════════════════════
    # HISTORY
    # ══════════════════════════════════════════════════════════════════
    def _show_history_tab(self):
        """Switch to history tab and load data."""
        self.notebook.select(4)
        self._load_history()

    def _load_history(self):
        """Load reading history from database."""
        # Clear existing
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        if not self._db_connected:
            self.history_detail_text.config(state='normal')
            self.history_detail_text.delete('1.0', 'end')
            self.history_detail_text.insert('1.0', '⚠ MySQL is not connected. Reading history is unavailable.')
            self.history_detail_text.config(state='disabled')
            return

        success, books = self.db.get_history()
        if not success:
            self.history_detail_text.config(state='normal')
            self.history_detail_text.delete('1.0', 'end')
            self.history_detail_text.insert('1.0', f'⚠ Failed to load history: {self.db.last_error}')
            self.history_detail_text.config(state='disabled')
            return

        if not books:
            self.history_detail_text.config(state='normal')
            self.history_detail_text.delete('1.0', 'end')
            self.history_detail_text.insert('1.0', 'No previous analyses found. Analyze a book to see it here.')
            self.history_detail_text.config(state='disabled')
            return

        for book in books:
            date_str = book['analyzed_at']
            if isinstance(date_str, datetime):
                date_str = date_str.strftime('%Y-%m-%d %H:%M')
            readability = book.get('readability_score')
            readability_str = f"{readability:.1f}" if readability is not None else "N/A"

            self.history_tree.insert('', 'end', values=(
                book['id'],
                book['title'],
                str(date_str),
                f"{book['word_count']:,}",
                readability_str,
                book['chapter_count'],
            ))

    def _view_history_details(self):
        """View details for a selected history entry."""
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a book from the history list.")
            return

        values = self.history_tree.item(selection[0], 'values')
        book_id = int(values[0])

        success, details = self.db.get_book_details(book_id)
        if not success or not details:
            messagebox.showerror("Error", "Could not load book details.")
            return

        book = details['book']
        chapters = details['chapters']
        topics = details['topics']

        # Format details text
        lines = []
        lines.append(f"═══ {book['title']} ═══\n")
        lines.append(f"File: {book['filename']}")
        lines.append(f"Analyzed: {book['analyzed_at']}")
        lines.append(f"Words: {book['word_count']:,}  |  Sentences: {book['sentence_count']:,}")
        rs = book.get('readability_score')
        lines.append(f"Readability (Flesch): {rs:.1f}" if rs else "Readability: N/A")
        lines.append(f"Chapters: {book['chapter_count']}\n")

        if chapters:
            lines.append("── Chapter Sentiment ──")
            for ch in chapters:
                lines.append(
                    f"  {ch['chapter_title']}: "
                    f"compound={ch['sentiment_compound']:.4f}, "
                    f"pos={ch['sentiment_positive']:.4f}, "
                    f"neu={ch['sentiment_neutral']:.4f}, "
                    f"neg={ch['sentiment_negative']:.4f}"
                )
            lines.append("")

        if topics:
            lines.append("── Topics ──")
            for t in topics:
                lines.append(f"  Topic {t['topic_number']}: {t['keywords']}")

        self.history_detail_text.config(state='normal')
        self.history_detail_text.delete('1.0', 'end')
        self.history_detail_text.insert('1.0', '\n'.join(lines))
        self.history_detail_text.config(state='disabled')

    def _delete_history_entry(self):
        """Delete a selected history entry."""
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a book to delete.")
            return

        values = self.history_tree.item(selection[0], 'values')
        book_id = int(values[0])
        title = values[1]

        if not messagebox.askyesno("Confirm Delete",
                                    f"Delete analysis for '{title}'?\nThis cannot be undone."):
            return

        success, msg = self.db.delete_book(book_id)
        if success:
            self._load_history()
            self._set_status(f"Deleted: {title}", 'Status.TLabel')
        else:
            messagebox.showerror("Error", msg)

    # ══════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════
    def _on_export(self, format: str):
        """Handle export button clicks."""
        if not self._analysis_results:
            messagebox.showinfo("Info", "No analysis results to export. Analyze a book first.")
            return

        results = self._analysis_results
        default_name = results['processed']['title'].replace(' ', '_')

        if format == 'json':
            filepath = filedialog.asksaveasfilename(
                title="Export JSON Report",
                defaultextension='.json',
                filetypes=[("JSON Files", "*.json")],
                initialfile=f"{default_name}_report.json"
            )
        else:
            filepath = filedialog.asksaveasfilename(
                title="Export CSV Report",
                defaultextension='.csv',
                filetypes=[("CSV Files", "*.csv")],
                initialfile=f"{default_name}_report.csv"
            )

        if not filepath:
            return

        try:
            if format == 'json':
                self.exporter.export_json(
                    filepath, results['processed'], results['readability'],
                    results['chapters'], results['chapter_sentiments'],
                    results['topic_results'], results['story_signature'],
                    results['interpretation']
                )
            else:
                self.exporter.export_csv(
                    filepath, results['processed'], results['readability'],
                    results['chapters'], results['chapter_sentiments'],
                    results['topic_results']
                )

            self._set_status(f"Report exported: {os.path.basename(filepath)}", 'Success.TLabel')
            messagebox.showinfo("Export Complete",
                                f"Report saved to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report:\n{str(e)}")

    def on_close(self):
        """Clean up resources on window close."""
        self._destroy_chart_canvases()
        self.db.close()
        self.root.destroy()
