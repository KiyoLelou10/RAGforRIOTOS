#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple

# === CONFIGURATION ===
EMBEDDINGS_FILE = Path("riot_examples_embeddings.jsonl")
MODEL_NAME = "all-mpnet-base-v2"  # Should match the model used for embedding
TOP_K = 8  # Number of results to return
PATH_BOOST = 0.1  # Boost score for path keyword matches

class RAGSearcher:
    def __init__(self, embeddings_file: Path, model_name: str):
        self.embeddings_file = embeddings_file
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings_matrix = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_embeddings(self):
        """Load chunks and embeddings from JSONL file"""
        print(f"Loading embeddings from {self.embeddings_file}")
        
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        
        chunks = []
        embeddings = []
        
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    chunks.append(record)
                    embeddings.append(np.array(record['embedding']))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
                except KeyError as e:
                    print(f"Warning: Missing key {e} on line {line_num}")
                    continue
        
        self.chunks = chunks
        self.embeddings_matrix = np.vstack(embeddings)
        print(f"Loaded {len(self.chunks)} chunks with embeddings")
        
    def extract_query_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query (remove common words)"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())
        
        # Filter out very common words (basic stopwords for technical content)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        return keywords
    
    def calculate_path_boost(self, query: str, file_path: str) -> float:
        """Calculate boost score based on query keywords found in file path"""
        keywords = self.extract_query_keywords(query)
        file_path_lower = file_path.lower()
        
        boost_score = 0.0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword in file_path_lower:
                boost_score += PATH_BOOST
                matched_keywords.append(keyword)
        
        return boost_score, matched_keywords
    
    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Search for most similar chunks to the query"""
        if not self.model or not self.chunks:
            raise ValueError("Model and embeddings must be loaded first")
        
        print(f"Searching for: '{query}'")
        
        # Embed the query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Calculate path boosts and combine scores
        results = []
        for i, chunk in enumerate(self.chunks):
            cosine_score = similarities[i]
            path_boost, matched_keywords = self.calculate_path_boost(query, chunk['file_path'])
            final_score = cosine_score + path_boost
            
            results.append({
                'chunk': chunk,
                'cosine_similarity': float(cosine_score),
                'path_boost': float(path_boost),
                'final_score': float(final_score),
                'matched_keywords': matched_keywords
            })
        
        # Sort by final score and return top k
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]
    
    def print_results(self, results: List[Dict], query: str, show_preview_only: bool = False):
        """Pretty print search results - now shows full chunks by default"""
        print(f"\n{'='*80}")
        print(f"TOP {len(results)} RESULTS FOR: '{query}'")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            print(f"\n[{i}] Score: {result['final_score']:.4f} " +
                  f"(cosine: {result['cosine_similarity']:.4f}, " +
                  f"path_boost: {result['path_boost']:.4f})")
            
            if result['matched_keywords']:
                print(f"🎯 Path keywords matched: {', '.join(result['matched_keywords'])}")
            
            print(f"📁 File: {chunk['file_path']}")
            print(f"🔖 Chunk: {chunk['chunk_id']}")
            
            if show_preview_only:
                # Show text preview (first 200 chars) - old behavior
                text_preview = chunk['text'][:200].replace('\n', ' ').strip()
                if len(chunk['text']) > 200:
                    text_preview += "..."
                print(f"📄 Preview: {text_preview}")
            else:
                # Show full text - new default behavior
                print(f"📄 Full Text:")
                print("-" * 40)
                print(chunk['text'])
                print("-" * 40)
            
            print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Search RIOT examples using semantic similarity")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, 
                       help=f"Number of results to return (default: {TOP_K})")
    parser.add_argument("-f", "--file", type=str, default=str(EMBEDDINGS_FILE),
                       help=f"Path to embeddings file (default: {EMBEDDINGS_FILE})")
    parser.add_argument("-m", "--model", type=str, default=MODEL_NAME,
                       help=f"Model name (default: {MODEL_NAME})")
    parser.add_argument("--preview-only", action="store_true",
                       help="Show only preview of results (first 200 chars) instead of full text")
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        searcher = RAGSearcher(Path(args.file), args.model)
        
        # Load model and embeddings
        searcher.load_model()
        searcher.load_embeddings()
        
        # Perform search
        results = searcher.search(args.query, args.top_k)
        
        # Print results (full text by default, preview only if requested)
        searcher.print_results(results, args.query, show_preview_only=args.preview_only)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())