#!/usr/bin/env python3
"""
RIOT OS RAG Systems GUI Application

A comprehensive GUI for accessing three different RIOT OS RAG query systems:
1. Standard RAG (ChromaDB-based documentation search)
2. Compressed RAG (Autoencoder-compressed embeddings)
3. Combined RAG (Documentation + Examples unified search)

Dependencies:
    pip install tkinter chromadb sentence-transformers torch numpy scikit-learn

Author: Assistant
License: MIT
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystemGUI:
    """GUI Application for RIOT OS RAG Systems"""

    def __init__(self, root):
        self.root = root
        self.root.title("RIOT OS RAG Query System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Variables
        self.current_rag_system = None
        self.rag_type = tk.StringVar(value="standard")
        self.query_var = tk.StringVar()
        self.is_processing = False

        # System-specific parameters
        self.standard_params = {
            'db_path': tk.StringVar(value="./riot_vector_db"),
            'collection': tk.StringVar(value="riot_docs"),
            'model': tk.StringVar(value="sentence-transformers/all-mpnet-base-v2"),
            'chunks': tk.IntVar(value=8),
            'prefer_module': tk.StringVar(value=""),
            'prefer_category': tk.StringVar(value=""),
        }

        self.compressed_params = {
            'chunks_file': tk.StringVar(value="compressed_chunks.json"),
            'model_file': tk.StringVar(value="autoencoder_model.pth"),
            'scaler_file': tk.StringVar(value="embedding_scaler.pkl"),
            'embedding_model': tk.StringVar(value="sentence-transformers/all-mpnet-base-v2"),
            'chunks': tk.IntVar(value=8),
            'prefer_module': tk.StringVar(value=""),
            'prefer_category': tk.StringVar(value=""),
        }

        self.combined_params = {
            'doc_db_path': tk.StringVar(value="./riot_vector_db"),
            'doc_collection': tk.StringVar(value="riot_docs"),
            'examples_file': tk.StringVar(value="./riot_examples_embeddings.jsonl"),
            'model': tk.StringVar(value="sentence-transformers/all-mpnet-base-v2"),
            'doc_chunks': tk.IntVar(value=8),
            'example_chunks': tk.IntVar(value=8),
            'prefer_module': tk.StringVar(value=""),
            'prefer_category': tk.StringVar(value=""),
        }

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_frame, text="RIOT OS RAG Query System",
                                font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))

        # Create notebook for different views
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Setup tabs
        self.setup_selection_tab()
        self.setup_parameters_tab()
        self.setup_results_tab()

        # Query input frame
        self.setup_query_frame(main_frame)

    def setup_selection_tab(self):
        """Setup the RAG system selection tab"""
        selection_frame = ttk.Frame(self.notebook)
        self.notebook.add(selection_frame, text="System Selection")

        # Create selection area
        selection_inner = ttk.LabelFrame(selection_frame, text="Choose RAG System", padding=20)
        selection_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Radio buttons for system selection
        systems = [
            ("standard", "Standard RAG", "ChromaDB-based documentation search with advanced reranking"),
            ("compressed", "Compressed RAG", "Autoencoder-compressed embeddings for memory efficiency"),
            ("combined", "Combined RAG", "Unified documentation + examples search system")
        ]

        for value, title, description in systems:
            frame = ttk.Frame(selection_inner)
            frame.pack(fill=tk.X, pady=10)

            radio = ttk.Radiobutton(frame, text=title, variable=self.rag_type,
                                    value=value, command=self.on_system_change)
            radio.pack(anchor=tk.W)

            desc_label = ttk.Label(frame, text=description, foreground='gray')
            desc_label.pack(anchor=tk.W, padx=(20, 0))

        # System info area
        self.info_frame = ttk.LabelFrame(selection_inner, text="System Information", padding=10)
        self.info_frame.pack(fill=tk.X, pady=(20, 0))

        self.info_text = tk.Text(self.info_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        info_scrollbar = ttk.Scrollbar(self.info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)

        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialize info
        self.update_system_info()

    def setup_parameters_tab(self):
        """Setup the parameters configuration tab"""
        params_frame = ttk.Frame(self.notebook)
        self.notebook.add(params_frame, text="Parameters")

        # Create scrollable frame
        canvas = tk.Canvas(params_frame)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Parameter frames for each system
        self.param_frames = {}

        # Standard RAG parameters
        self.param_frames['standard'] = self.create_standard_params(scrollable_frame)

        # Compressed RAG parameters
        self.param_frames['compressed'] = self.create_compressed_params(scrollable_frame)

        # Combined RAG parameters
        self.param_frames['combined'] = self.create_combined_params(scrollable_frame)

        # Show initial parameters
        self.show_current_params()

    def create_standard_params(self, parent):
        """Create parameter widgets for Standard RAG"""
        frame = ttk.LabelFrame(parent, text="Standard RAG Parameters", padding=15)

        # Database path
        ttk.Label(frame, text="Database Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        db_entry = ttk.Entry(frame, textvariable=self.standard_params['db_path'], width=40)
        db_entry.grid(row=0, column=1, sticky=tk.EW, padx=(10, 5), pady=5)
        ttk.Button(frame, text="Browse",
                   command=lambda: self.browse_directory(self.standard_params['db_path'])).grid(row=0, column=2, pady=5)

        # Collection name
        ttk.Label(frame, text="Collection:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.standard_params['collection'], width=40).grid(row=1, column=1, sticky=tk.EW,
                                                                                         padx=(10, 0), pady=5)

        # Model name
        ttk.Label(frame, text="Model:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.standard_params['model'], width=40).grid(row=2, column=1, sticky=tk.EW,
                                                                                    padx=(10, 0), pady=5)

        # Number of chunks
        ttk.Label(frame, text="Chunks:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(frame, textvariable=self.standard_params['chunks'], from_=1, to=50, width=10).grid(row=3, column=1,
                                                                                                       sticky=tk.W,
                                                                                                       padx=(10, 0),
                                                                                                       pady=5)

        # Preferences
        ttk.Label(frame, text="Prefer Module:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.standard_params['prefer_module'], width=40).grid(row=4, column=1,
                                                                                            sticky=tk.EW, padx=(10, 0),
                                                                                            pady=5)

        ttk.Label(frame, text="Prefer Category:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.standard_params['prefer_category'], width=40).grid(row=5, column=1,
                                                                                              sticky=tk.EW,
                                                                                              padx=(10, 0), pady=5)

        frame.columnconfigure(1, weight=1)
        return frame

    def create_compressed_params(self, parent):
        """Create parameter widgets for Compressed RAG"""
        frame = ttk.LabelFrame(parent, text="Compressed RAG Parameters", padding=15)

        # Chunks file
        ttk.Label(frame, text="Chunks File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        chunks_entry = ttk.Entry(frame, textvariable=self.compressed_params['chunks_file'], width=40)
        chunks_entry.grid(row=0, column=1, sticky=tk.EW, padx=(10, 5), pady=5)
        ttk.Button(frame, text="Browse",
                   command=lambda: self.browse_file(self.compressed_params['chunks_file'], "JSON files",
                                                    "*.json")).grid(row=0, column=2, pady=5)

        # Model file
        ttk.Label(frame, text="Model File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        model_entry = ttk.Entry(frame, textvariable=self.compressed_params['model_file'], width=40)
        model_entry.grid(row=1, column=1, sticky=tk.EW, padx=(10, 5), pady=5)
        ttk.Button(frame, text="Browse",
                   command=lambda: self.browse_file(self.compressed_params['model_file'], "PyTorch files",
                                                    "*.pth")).grid(row=1, column=2, pady=5)

        # Scaler file
        ttk.Label(frame, text="Scaler File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        scaler_entry = ttk.Entry(frame, textvariable=self.compressed_params['scaler_file'], width=40)
        scaler_entry.grid(row=2, column=1, sticky=tk.EW, padx=(10, 5), pady=5)
        ttk.Button(frame, text="Browse",
                   command=lambda: self.browse_file(self.compressed_params['scaler_file'], "Pickle files",
                                                    "*.pkl")).grid(row=2, column=2, pady=5)

        # Embedding model
        ttk.Label(frame, text="Embedding Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.compressed_params['embedding_model'], width=40).grid(row=3, column=1,
                                                                                                sticky=tk.EW,
                                                                                                padx=(10, 0), pady=5)

        # Number of chunks
        ttk.Label(frame, text="Chunks:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(frame, textvariable=self.compressed_params['chunks'], from_=1, to=50, width=10).grid(row=4,
                                                                                                         column=1,
                                                                                                         sticky=tk.W,
                                                                                                         padx=(10, 0),
                                                                                                         pady=5)

        # Preferences
        ttk.Label(frame, text="Prefer Module:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.compressed_params['prefer_module'], width=40).grid(row=5, column=1,
                                                                                              sticky=tk.EW,
                                                                                              padx=(10, 0), pady=5)

        ttk.Label(frame, text="Prefer Category:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.compressed_params['prefer_category'], width=40).grid(row=6, column=1,
                                                                                                sticky=tk.EW,
                                                                                                padx=(10, 0), pady=5)

        frame.columnconfigure(1, weight=1)
        return frame

    def create_combined_params(self, parent):
        """Create parameter widgets for Combined RAG"""
        frame = ttk.LabelFrame(parent, text="Combined RAG Parameters", padding=15)

        # Doc database path
        ttk.Label(frame, text="Doc DB Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        doc_db_entry = ttk.Entry(frame, textvariable=self.combined_params['doc_db_path'], width=40)
        doc_db_entry.grid(row=0, column=1, sticky=tk.EW, padx=(10, 5), pady=5)
        ttk.Button(frame, text="Browse",
                   command=lambda: self.browse_directory(self.combined_params['doc_db_path'])).grid(row=0, column=2,
                                                                                                    pady=5)

        # Doc collection
        ttk.Label(frame, text="Doc Collection:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.combined_params['doc_collection'], width=40).grid(row=1, column=1,
                                                                                             sticky=tk.EW, padx=(10, 0),
                                                                                             pady=5)

        # Examples file
        ttk.Label(frame, text="Examples File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        examples_entry = ttk.Entry(frame, textvariable=self.combined_params['examples_file'], width=40)
        examples_entry.grid(row=2, column=1, sticky=tk.EW, padx=(10, 5), pady=5)
        ttk.Button(frame, text="Browse",
                   command=lambda: self.browse_file(self.combined_params['examples_file'], "JSONL files",
                                                    "*.jsonl")).grid(row=2, column=2, pady=5)

        # Model
        ttk.Label(frame, text="Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.combined_params['model'], width=40).grid(row=3, column=1, sticky=tk.EW,
                                                                                    padx=(10, 0), pady=5)

        # Doc chunks
        ttk.Label(frame, text="Doc Chunks:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(frame, textvariable=self.combined_params['doc_chunks'], from_=1, to=50, width=10).grid(row=4,
                                                                                                           column=1,
                                                                                                           sticky=tk.W,
                                                                                                           padx=(10, 0),
                                                                                                           pady=5)

        # Example chunks
        ttk.Label(frame, text="Example Chunks:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(frame, textvariable=self.combined_params['example_chunks'], from_=1, to=50, width=10).grid(row=5,
                                                                                                               column=1,
                                                                                                               sticky=tk.W,
                                                                                                               padx=(
                                                                                                               10, 0),
                                                                                                               pady=5)

        # Preferences
        ttk.Label(frame, text="Prefer Module:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.combined_params['prefer_module'], width=40).grid(row=6, column=1,
                                                                                            sticky=tk.EW, padx=(10, 0),
                                                                                            pady=5)

        ttk.Label(frame, text="Prefer Category:").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.combined_params['prefer_category'], width=40).grid(row=7, column=1,
                                                                                              sticky=tk.EW,
                                                                                              padx=(10, 0), pady=5)

        frame.columnconfigure(1, weight=1)
        return frame

    def setup_results_tab(self):
        """Setup the results display tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")

        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD,
                                                      state=tk.DISABLED, font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Results control frame
        control_frame = ttk.Frame(results_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(control_frame, text="Save Results",
                   command=self.save_results).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Clear Results",
                   command=self.clear_results).pack(side=tk.LEFT, padx=(10, 0))

    def setup_query_frame(self, parent):
        """Setup the query input frame"""
        query_frame = ttk.LabelFrame(parent, text="Query Input", padding=10)
        query_frame.pack(fill=tk.X, pady=(10, 0))

        # Query entry
        self.query_entry = ttk.Entry(query_frame, textvariable=self.query_var,
                                     font=('Arial', 11))
        self.query_entry.pack(fill=tk.X, pady=(0, 10))
        self.query_entry.bind('<Return>', lambda e: self.execute_query())

        # Control buttons
        button_frame = ttk.Frame(query_frame)
        button_frame.pack(fill=tk.X)

        self.execute_button = ttk.Button(button_frame, text="Execute Query",
                                         command=self.execute_query)
        self.execute_button.pack(side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        ttk.Button(button_frame, text="Load System Stats",
                   command=self.load_system_stats).pack(side=tk.RIGHT)

    def on_system_change(self):
        """Handle RAG system selection change"""
        self.show_current_params()
        self.update_system_info()

    def show_current_params(self):
        """Show parameters for current system"""
        # Hide all parameter frames
        for frame in self.param_frames.values():
            frame.pack_forget()

        # Show current system's parameters
        current_frame = self.param_frames.get(self.rag_type.get())
        if current_frame:
            current_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_system_info(self):
        """Update system information display"""
        system_info = {
            'standard': """Standard RAG System (ChromaDB-based)

Features:
• Advanced semantic search using ChromaDB vector database
• Hybrid reranking with similarity, diversity, and preference scoring
• Support for module and category preferences
• Configurable chunk retrieval with intelligent diversity promotion

Requirements:
• ChromaDB database with RIOT OS documentation
• Sentence transformer model for embeddings
• Valid collection with pre-indexed documents

Best for: Comprehensive documentation search with advanced filtering""",

            'compressed': """Compressed RAG System (Autoencoder-based)

Features:
• Memory-efficient compressed embeddings (768→256 dimensions)
• Trained autoencoder for embedding compression
• Fast similarity search with reduced memory footprint
• Maintains high semantic quality with compression

Requirements:
• Pre-trained autoencoder model (.pth file)
• Embedding scaler (.pkl file)  
• Compressed chunks JSON file with embeddings

Best for: Large-scale deployment with memory constraints""",

            'combined': """Combined RAG System (Documentation + Examples)

Features:
• Unified search across documentation and example code
• Hierarchical result prioritization (docs primary, examples supplementary)
• Path-based keyword boosting for example relevance
• Comprehensive context with both theory and practice

Requirements:
• ChromaDB database for documentation
• JSONL file with example embeddings
• Both documentation and example collections

Best for: Complete learning experience with theory and practice"""
        }

        info = system_info.get(self.rag_type.get(), "No information available")

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)

    def browse_directory(self, var):
        """Browse for directory"""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)

    def browse_file(self, var, file_type, pattern):
        """Browse for file"""
        filename = filedialog.askopenfilename(
            title=f"Select {file_type}",
            filetypes=[(file_type, pattern), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)

    def get_current_params(self) -> Dict[str, Any]:
        """Get parameters for current system"""
        system = self.rag_type.get()

        if system == 'standard':
            return {
                'db_path': self.standard_params['db_path'].get(),
                'collection': self.standard_params['collection'].get(),
                'model': self.standard_params['model'].get(),
                'chunks': self.standard_params['chunks'].get(),
                'prefer_module': self.standard_params['prefer_module'].get() or None,
                'prefer_category': self.standard_params['prefer_category'].get() or None,
            }
        elif system == 'compressed':
            return {
                'chunks_file': self.compressed_params['chunks_file'].get(),
                'model_file': self.compressed_params['model_file'].get(),
                'scaler_file': self.compressed_params['scaler_file'].get(),
                'embedding_model': self.compressed_params['embedding_model'].get(),
                'chunks': self.compressed_params['chunks'].get(),
                'prefer_module': self.compressed_params['prefer_module'].get() or None,
                'prefer_category': self.compressed_params['prefer_category'].get() or None,
            }
        elif system == 'combined':
            return {
                'doc_db_path': self.combined_params['doc_db_path'].get(),
                'doc_collection': self.combined_params['doc_collection'].get(),
                'examples_file': self.combined_params['examples_file'].get(),
                'model': self.combined_params['model'].get(),
                'doc_chunks': self.combined_params['doc_chunks'].get(),
                'example_chunks': self.combined_params['example_chunks'].get(),
                'prefer_module': self.combined_params['prefer_module'].get() or None,
                'prefer_category': self.combined_params['prefer_category'].get() or None,
            }

        return {}

    def initialize_rag_system(self):
        """Initialize the selected RAG system"""
        system_type = self.rag_type.get()
        params = self.get_current_params()

        try:
            if system_type == 'standard':
                # Import and initialize standard RAG
                from RIORDocuRAGRequest2 import RIOTRAGQuerySystem
                self.current_rag_system = RIOTRAGQuerySystem(
                    db_path=params['db_path'],
                    collection_name=params['collection'],
                    model_name=params['model']
                )

            elif system_type == 'compressed':
                # Import and initialize compressed RAG
                from RIORDocuRAGRequestCompressed import RIOTCompressedSearchSystem
                self.current_rag_system = RIOTCompressedSearchSystem(
                    chunks_file=params['chunks_file'],
                    model_file=params['model_file'],
                    scaler_file=params['scaler_file'],
                    embedding_model_name=params['embedding_model']
                )

            elif system_type == 'combined':
                # Import and initialize combined RAG
                from RIOTRequestCombined import RIOTUnifiedRAGSystem
                self.current_rag_system = RIOTUnifiedRAGSystem(
                    doc_db_path=params['doc_db_path'],
                    doc_collection_name=params['doc_collection'],
                    examples_embeddings_file=params['examples_file'],
                    model_name=params['model']
                )

            return True

        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize {system_type} RAG system:\n\n{str(e)}")
            return False

    def execute_query(self):
        """Execute the query using the selected RAG system"""
        if self.is_processing:
            return

        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("No Query", "Please enter a query.")
            return

        # Start processing in separate thread
        self.is_processing = True
        self.execute_button.config(state=tk.DISABLED)
        self.progress_bar.start()

        thread = threading.Thread(target=self._execute_query_thread, args=(query,))
        thread.daemon = True
        thread.start()

    def _execute_query_thread(self, query: str):
        """Execute query in separate thread"""
        try:
            # Initialize system if needed
            if not self.current_rag_system:
                if not self.initialize_rag_system():
                    return

            system_type = self.rag_type.get()
            params = self.get_current_params()

            # Execute query based on system type
            if system_type == 'standard':
                result = self.current_rag_system.generate_query_response(
                    query,
                    n_chunks=params['chunks'],
                    prefer_module=params['prefer_module'],
                    prefer_category=params['prefer_category']
                )

            elif system_type == 'compressed':
                result = self.current_rag_system.generate_query_response(
                    query,
                    n_chunks=params['chunks'],
                    prefer_module=params['prefer_module'],
                    prefer_category=params['prefer_category']
                )

            elif system_type == 'combined':
                result = self.current_rag_system.generate_unified_response(
                    query,
                    doc_chunks=params['doc_chunks'],
                    example_chunks=params['example_chunks'],
                    prefer_module=params['prefer_module'],
                    prefer_category=params['prefer_category']
                )
            else:
                result = "Unknown system type"

            # Update UI in main thread
            self.root.after(0, self._display_results, query, result)

        except Exception as e:
            error_msg = f"Query execution failed:\n\n{str(e)}"
            self.root.after(0, self._show_error, error_msg)

        finally:
            # Re-enable UI in main thread
            self.root.after(0, self._finish_processing)

    def _display_results(self, query: str, result: str):
        """Display query results in main thread"""
        # Switch to results tab
        self.notebook.select(2)

        # Format and display results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n{'=' * 80}\nQuery executed at {timestamp}\nSystem: {self.rag_type.get().title()} RAG\nQuery: {query}\n{'=' * 80}\n\n"

        self.results_text.config(state=tk.NORMAL)
        if self.results_text.get(1.0, tk.END).strip():
            self.results_text.insert(tk.END, "\n\n")
        self.results_text.insert(tk.END, header + result)
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

    def _show_error(self, error_msg: str):
        """Show error message in main thread"""
        messagebox.showerror("Query Error", error_msg)

    def _finish_processing(self):
        """Finish processing and re-enable UI"""
        self.is_processing = False
        self.execute_button.config(state=tk.NORMAL)
        self.progress_bar.stop()

    def load_system_stats(self):
        """Load and display system statistics"""
        if not self.current_rag_system:
            if not self.initialize_rag_system():
                return

        try:
            stats = self.current_rag_system.get_system_stats()

            # Format stats nicely
            stats_text = f"\n{'=' * 60}\nSystem Statistics - {self.rag_type.get().title()} RAG\n{'=' * 60}\n\n"

            for key, value in stats.items():
                stats_text += f"{key.replace('_', ' ').title()}: {value}\n"

            # Switch to results tab and display
            self.notebook.select(2)
            self.results_text.config(state=tk.NORMAL)
            if self.results_text.get(1.0, tk.END).strip():
                self.results_text.insert(tk.END, "\n\n")
            self.results_text.insert(tk.END, stats_text)
            self.results_text.see(tk.END)
            self.results_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Stats Error", f"Failed to load system statistics:\n\n{str(e)}")

    def save_results(self):
        """Save results to file"""
        content = self.results_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("No Results", "No results to save.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results:\n\n{str(e)}")

    def clear_results(self):
        """Clear the results display"""
        if messagebox.askyesno("Clear Results", "Are you sure you want to clear all results?"):
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.config(state=tk.DISABLED)


class RAGSystemImporter:
    """Helper class to handle dynamic imports of RAG systems"""

    @staticmethod
    def check_file_exists(filename: str) -> bool:
        """Check if a Python file exists"""
        return Path(filename).exists()

    @staticmethod
    def validate_system_files():
        """Validate that all required system files exist"""
        required_files = [
            "RIORDocuRAGRequest2.py",
            "RIORDocuRAGRequestCompressed.py",
            "RIOTRequestCombined.py"
        ]

        missing_files = []
        for filename in required_files:
            if not RAGSystemImporter.check_file_exists(filename):
                missing_files.append(filename)

        return missing_files


def show_welcome_dialog():
    """Show welcome dialog with system information"""
    dialog = tk.Toplevel()
    dialog.title("Welcome to RIOT OS RAG System")
    dialog.geometry("600x400")
    dialog.resizable(False, False)

    # Make it modal
    dialog.transient()
    dialog.grab_set()

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
    y = (dialog.winfo_screenheight() // 2) - (400 // 2)
    dialog.geometry(f"600x400+{x}+{y}")

    # Welcome content
    welcome_text = """
Welcome to the RIOT OS RAG Query System!

This application provides access to three different RAG (Retrieval-Augmented Generation) systems for querying RIOT OS documentation:

🔍 Standard RAG
   • ChromaDB-based documentation search
   • Advanced reranking with semantic similarity
   • Best for comprehensive documentation queries

⚡ Compressed RAG  
   • Memory-efficient autoencoder-compressed embeddings
   • Fast similarity search with reduced footprint
   • Best for large-scale deployment scenarios

🔀 Combined RAG
   • Unified documentation + examples search
   • Hierarchical prioritization of results
   • Best for complete learning experience

Getting Started:
1. Select your preferred RAG system from the "System Selection" tab
2. Configure parameters in the "Parameters" tab
3. Enter your query in the input field at the bottom
4. View results in the "Results" tab

Requirements:
Make sure you have the following files in your directory:
• RIORDocuRAGRequest2.py (Standard RAG)
• RIORDocuRAGRequestCompressed.py (Compressed RAG)  
• RIOTRequestCombined.py (Combined RAG)

Ready to explore RIOT OS documentation with AI assistance!
    """

    text_widget = tk.Text(dialog, wrap=tk.WORD, padx=20, pady=20,
                          font=('Arial', 11), state=tk.DISABLED)
    text_widget.pack(fill=tk.BOTH, expand=True)

    # Insert welcome text
    text_widget.config(state=tk.NORMAL)
    text_widget.insert(tk.END, welcome_text.strip())
    text_widget.config(state=tk.DISABLED)

    # OK button
    button_frame = ttk.Frame(dialog)
    button_frame.pack(pady=20)

    ttk.Button(button_frame, text="Get Started",
               command=dialog.destroy).pack()

    # Wait for dialog to close
    dialog.wait_window()


def main():
    """Main application entry point"""
    # Validate system files
    missing_files = RAGSystemImporter.validate_system_files()

    if missing_files:
        root = tk.Tk()
        root.withdraw()  # Hide main window

        error_msg = ("Missing required RAG system files:\n\n" +
                     "\n".join(f"• {file}" for file in missing_files) +
                     "\n\nPlease ensure all RAG system files are in the same directory as this GUI application.")

        messagebox.showerror("Missing Files", error_msg)
        return 1

    # Create main application
    root = tk.Tk()

    try:
        app = RAGSystemGUI(root)

        # Show welcome dialog
        root.after(500, show_welcome_dialog)  # Show after main window loads

        # Start main loop
        root.mainloop()

    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application:\n\n{str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())