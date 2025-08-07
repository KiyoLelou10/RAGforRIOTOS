#!/usr/bin/env python3
"""
RIOT OS RAG Systems Web Server

A Flask-based web server for accessing three different RIOT OS RAG query systems:
1. Standard RAG (ChromaDB-based documentation search)
2. Compressed RAG (Autoencoder-compressed embeddings)
3. Combined RAG (Documentation + Examples unified search)

Dependencies:
    pip install flask chromadb sentence-transformers torch numpy scikit-learn flask-cors

Author: Assistant
License: MIT
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import threading
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class RAGSystemManager:
    """Manages the RAG systems and their instances"""

    def __init__(self):
        self.systems = {
            'standard': None,
            'compressed': None,
            'combined': None
        }
        self.system_lock = threading.Lock()

    def initialize_system(self, system_type: str, params: Dict[str, Any]) -> bool:
        """Initialize a RAG system with given parameters"""
        with self.system_lock:
            try:
                if system_type == 'standard':
                    from RIORDocuRAGRequest2 import RIOTRAGQuerySystem
                    self.systems['standard'] = RIOTRAGQuerySystem(
                        db_path=params['db_path'],
                        collection_name=params['collection'],
                        model_name=params['model']
                    )

                elif system_type == 'compressed':
                    from RIORDocuRAGRequestCompressed import RIOTCompressedSearchSystem
                    self.systems['compressed'] = RIOTCompressedSearchSystem(
                        chunks_file=params['chunks_file'],
                        model_file=params['model_file'],
                        scaler_file=params['scaler_file'],
                        embedding_model_name=params['embedding_model']
                    )

                elif system_type == 'combined':
                    from RIOTRequestCombined import RIOTUnifiedRAGSystem
                    self.systems['combined'] = RIOTUnifiedRAGSystem(
                        doc_db_path=params['doc_db_path'],
                        doc_collection_name=params['doc_collection'],
                        examples_embeddings_file=params['examples_file'],
                        model_name=params['model']
                    )

                return True

            except Exception as e:
                logger.error(f"Failed to initialize {system_type} system: {str(e)}")
                return False

    def get_system(self, system_type: str):
        """Get initialized system or None"""
        return self.systems.get(system_type)

    def query_system(self, system_type: str, query: str, params: Dict[str, Any]) -> str:
        """Execute query on specified system"""
        system = self.get_system(system_type)
        if not system:
            raise Exception(f"System {system_type} not initialized")

        if system_type == 'standard':
            return system.generate_query_response(
                query,
                n_chunks=params.get('chunks', 8),
                prefer_module=params.get('prefer_module'),
                prefer_category=params.get('prefer_category')
            )

        elif system_type == 'compressed':
            return system.generate_query_response(
                query,
                n_chunks=params.get('chunks', 8),
                prefer_module=params.get('prefer_module'),
                prefer_category=params.get('prefer_category')
            )

        elif system_type == 'combined':
            return system.generate_unified_response(
                query,
                doc_chunks=params.get('doc_chunks', 8),
                example_chunks=params.get('example_chunks', 8),
                prefer_module=params.get('prefer_module'),
                prefer_category=params.get('prefer_category')
            )

        raise Exception(f"Unknown system type: {system_type}")

    def get_system_stats(self, system_type: str) -> Dict[str, Any]:
        """Get system statistics"""
        system = self.get_system(system_type)
        if not system:
            raise Exception(f"System {system_type} not initialized")

        return system.get_system_stats()


# Global RAG system manager
rag_manager = RAGSystemManager()


@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')


@app.route('/api/validate-files', methods=['POST'])
def validate_files():
    """Validate that required RAG system files exist"""
    required_files = [
        "RIORDocuRAGRequest2.py",
        "RIORDocuRAGRequestCompressed.py",
        "RIOTRequestCombined.py"
    ]

    missing_files = []
    for filename in required_files:
        if not Path(filename).exists():
            missing_files.append(filename)

    return jsonify({
        'valid': len(missing_files) == 0,
        'missing_files': missing_files
    })


@app.route('/api/initialize-system', methods=['POST'])
def initialize_system():
    """Initialize a RAG system"""
    try:
        data = request.json
        system_type = data['system_type']
        params = data['parameters']

        success = rag_manager.initialize_system(system_type, params)

        return jsonify({
            'success': success,
            'message': f"{'Successfully' if success else 'Failed to'} initialize {system_type} system"
        })

    except Exception as e:
        logger.error(f"System initialization error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        }), 500


@app.route('/api/query', methods=['POST'])
def execute_query():
    """Execute a query on the specified RAG system"""
    try:
        data = request.json
        system_type = data['system_type']
        query = data['query']
        params = data['parameters']

        # Initialize system if not already done
        if not rag_manager.get_system(system_type):
            if not rag_manager.initialize_system(system_type, params):
                raise Exception(f"Failed to initialize {system_type} system")

        # Execute query
        result = rag_manager.query_system(system_type, query, params)

        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'system_type': system_type
        })

    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        }), 500


@app.route('/api/system-stats', methods=['POST'])
def get_system_stats():
    """Get system statistics"""
    try:
        data = request.json
        system_type = data['system_type']
        params = data.get('parameters', {})

        # Initialize system if not already done
        if not rag_manager.get_system(system_type):
            if not rag_manager.initialize_system(system_type, params):
                raise Exception(f"Failed to initialize {system_type} system")

        stats = rag_manager.get_system_stats(system_type)

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        logger.error(f"Stats retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        }), 500


@app.route('/api/check-file-exists', methods=['POST'])
def check_file_exists():
    """Check if a file exists"""
    try:
        data = request.json
        filepath = data['filepath']
        exists = Path(filepath).exists()

        return jsonify({
            'exists': exists,
            'filepath': filepath
        })

    except Exception as e:
        return jsonify({
            'exists': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)

    print("RIOT OS RAG System Server")
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("Make sure you have the required RAG system files in this directory:")
    print("• RIORDocuRAGRequest2.py")
    print("• RIORDocuRAGRequestCompressed.py")
    print("• RIOTRequestCombined.py")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)