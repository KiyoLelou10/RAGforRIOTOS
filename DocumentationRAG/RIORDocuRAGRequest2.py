#!/usr/bin/env python3
"""
RIOT OS Documentation RAG Query System

This script retrieves the most relevant RIOT OS documentation chunks for a user query
and formats them into an optimized prompt for Claude or other LLM models.

Dependencies:
    pip install chromadb sentence-transformers torch numpy

Usage:
    python riot_query.py "How do I configure GPIO pins in RIOT?"
    python riot_query.py "RIOT networking stack usage" --module sys
    python riot_query.py "Timer configuration" --output prompt.txt

Author: Assistant
License: MIT
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Vector database and embedding imports
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RIOTRAGQuerySystem:
    """RAG query system for RIOT OS documentation with advanced retrieval and prompt optimization."""

    def __init__(self, db_path: str = "./riot_vector_db",
                 collection_name: str = "riot_docs",
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the RAG query system."""
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.model_name = model_name

        # Initialize components
        self.model = self._load_embedding_model()
        self.chroma_client = self._initialize_chromadb()
        self.collection = self._load_collection()

        logger.info(f"RAG system initialized with collection: {collection_name}")

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence transformer model (same as used for embedding)."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded embedding model: {self.model_name} on {device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback model
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _initialize_chromadb(self) -> chromadb.Client:
        """Initialize ChromaDB client."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database path not found: {self.db_path}")

        return chromadb.PersistentClient(path=str(self.db_path))

    def _load_collection(self) -> chromadb.Collection:
        """Load the collection from ChromaDB."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            logger.info(f"Loaded collection '{self.collection_name}' with {count} documents")
            return collection
        except Exception as e:
            raise RuntimeError(f"Failed to load collection '{self.collection_name}': {e}")

    def _enhance_query(self, query: str) -> str:
        """Enhance query with RIOT-specific context (same as embedding enhancement)."""
        return f"RIOT OS documentation query: {query}"

    def _calculate_diversity_score(self, results: Dict[str, Any]) -> List[float]:
        """Calculate diversity scores to promote varied results."""
        if not results['metadatas'] or not results['metadatas'][0]:
            return [0.0] * len(results['documents'][0])

        metadatas = results['metadatas'][0]
        diversity_scores = []

        seen_modules = set()
        seen_files = set()
        seen_categories = set()

        for metadata in metadatas:
            score = 0.0

            # Reward new modules
            module = metadata.get('module', '')
            if module and module not in seen_modules:
                score += 0.3
                seen_modules.add(module)

            # Reward new files
            file_name = metadata.get('file_name', '')
            if file_name and file_name not in seen_files:
                score += 0.2
                seen_files.add(file_name)

            # Reward new categories
            category = metadata.get('category', '')
            if category and category not in seen_categories:
                score += 0.1
                seen_categories.add(category)

            diversity_scores.append(score)

        return diversity_scores

    def _rerank_results(self, results: Dict[str, Any],
                        query: str,
                        prefer_module: Optional[str] = None,
                        prefer_category: Optional[str] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        """Rerank results using hybrid scoring (similarity + diversity + preferences)."""
        if not results['documents'] or not results['documents'][0]:
            return []

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Calculate similarity scores (1 - distance)
        similarity_scores = [1 - d for d in distances]

        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_score(results)

        # Calculate preference scores
        query_words = [w for w in query.lower().split() if w]

        preference_scores = []
        for metadata in metadatas:
            score = 0.0

            # — existing boosts —
            if prefer_module and metadata.get('module', '').lower() == prefer_module.lower():
                score += 0.4
            if prefer_category and metadata.get('category', '').lower() == prefer_category.lower():
                score += 0.3
            if metadata.get('has_functions') == 'true':
                score += 0.1
            if 'example' in metadata.get('file_path', '').lower():
                score += 0.2

            # — new: bump for query-word matches —
            fp = metadata.get('file_path', '').lower()
            mod = metadata.get('module', '').lower()
            cat = metadata.get('category', '').lower()

            for w in query_words:
                if len(w) > 3:
                    if w in fp or w == mod or w == cat:
                        score += 0.1

            preference_scores.append(score)
        #preference_scores = []
        #for metadata in metadatas:
        #    score = 0.0

            # Module preference
         #   if prefer_module and metadata.get('module', '').lower() == prefer_module.lower():
          #      score += 0.4

            # Category preference
           # if prefer_category and metadata.get('category', '').lower() == prefer_category.lower():
            #    score += 0.3

            # Boost for API documentation
            #if metadata.get('has_functions') == 'true':
            #    score += 0.1

            # Boost for example code
            #if 'example' in metadata.get('file_path', '').lower():
             #   score += 0.2

           # preference_scores.append(score)

        # Combine scores with weights
        combined_scores = []
        for i in range(len(documents)):
            combined_score = (
                    0.6 * similarity_scores[i] +  # Primary: semantic similarity
                    0.2 * diversity_scores[i] +  # Secondary: diversity
                    0.2 * preference_scores[i]  # Tertiary: preferences
            )
            combined_scores.append(combined_score)

        # Create ranked results
        ranked_results = []
        for i, score in enumerate(combined_scores):
            ranked_results.append((documents[i], metadatas[i], score))

        # Sort by combined score (descending)
        ranked_results.sort(key=lambda x: x[2], reverse=True)

        return ranked_results

    def search_chunks(self, query: str,
                      n_results: int = 16,  # Get more initially for reranking
                      prefer_module: Optional[str] = None,
                      prefer_category: Optional[str] = None,
                      where: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for relevant chunks with advanced reranking."""

        # Enhance query with RIOT context
        enhanced_query = self._enhance_query(query)

        # Generate query embedding
        query_embedding = self.model.encode(
            [enhanced_query],
            normalize_embeddings=True
        )

        # Search parameters
        search_kwargs = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        # Add metadata filtering if specified
        if where:
            search_kwargs["where"] = where
        elif prefer_module:
            # Soft filter: prefer module but don't exclude others
            pass

        # Perform search
        results = self.collection.query(**search_kwargs)

        # Rerank results
        ranked_results = self._rerank_results(
            results, query, prefer_module, prefer_category
        )

        logger.info(f"Retrieved and reranked {len(ranked_results)} chunks for query: {query}")
        return ranked_results

    def get_top_chunks(self, query: str,
                       n_chunks: int = 8,
                       prefer_module: Optional[str] = None,
                       prefer_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the top N most relevant chunks."""

        # Search with larger initial set for better reranking
        search_results = self.search_chunks(
            query,
            n_results=min(n_chunks * 2, 32),  # Get 2x for better selection
            prefer_module=prefer_module,
            prefer_category=prefer_category
        )

        # Select top N chunks
        top_chunks = []
        for i, (document, metadata, score) in enumerate(search_results[:n_chunks]):
            chunk_info = {
                'rank': i + 1,
                'content': document,
                'metadata': metadata,
                'relevance_score': score,
                'file_path': metadata.get('file_path', ''),
                'module': metadata.get('module', ''),
                'category': metadata.get('category', ''),
                'file_name': metadata.get('file_name', ''),
                'section': metadata.get('section', ''),
                'tags': metadata.get('tags', '')
            }
            top_chunks.append(chunk_info)

        return top_chunks

    def format_claude_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Format the retrieved chunks into an optimized prompt for Claude."""

        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build the prompt
        prompt = f"""You are a RIOT OS expert assistant. Use the following documentation chunks to answer the user's question about RIOT OS.

**User Question:** {query}

**RIOT OS Documentation Context:**
The following {len(chunks)} chunks were retrieved from the official RIOT OS documentation based on semantic similarity and relevance to your question:

"""

        # Add each chunk with proper formatting
        for chunk in chunks:
            prompt += f"""
---
**Chunk {chunk['rank']}** (Relevance: {chunk['relevance_score']:.3f})
**Module:** {chunk['module']}
**Category:** {chunk['category']}
**File:** {chunk['file_name']}
**Path:** {chunk['file_path']}
**Section:** {chunk['section']}

```
{chunk['content']}
```

"""

        # Add instructions for Claude
        prompt += """
---

**Instructions:**
1. Use the provided RIOT OS documentation chunks to answer the user's question comprehensively
2. Focus on practical, actionable information from the documentation
3. Reference specific modules, functions, or APIs mentioned in the chunks when relevant
4. If the chunks don't contain sufficient information, clearly state what information is missing
5. Provide code examples or configuration snippets when available in the chunks
6. Maintain technical accuracy and use RIOT OS terminology correctly
7. Structure your response clearly with headers and bullet points when appropriate

**Response Guidelines:**
- Start with a direct answer to the user's question
- Provide step-by-step instructions when applicable
- Include relevant code examples from the documentation
- Mention any prerequisites or dependencies
- Reference specific file paths or modules when helpful
- End with related concepts or next steps if appropriate

Please provide a comprehensive answer based on the RIOT OS documentation provided above.
"""

        return prompt

    def generate_query_response(self, query: str,
                                n_chunks: int = 8,
                                prefer_module: Optional[str] = None,
                                prefer_category: Optional[str] = None,
                                output_file: Optional[str] = None) -> str:
        """Generate a complete RAG response for the query."""

        logger.info(f"Processing query: {query}")

        # Retrieve relevant chunks
        chunks = self.get_top_chunks(
            query,
            n_chunks=n_chunks,
            prefer_module=prefer_module,
            prefer_category=prefer_category
        )

        if not chunks:
            return f"No relevant RIOT OS documentation found for query: {query}"

        # Format the prompt
        formatted_prompt = self.format_claude_prompt(query, chunks)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_prompt)
            logger.info(f"Prompt saved to: {output_file}")

        # Log retrieval summary
        logger.info(f"Retrieved {len(chunks)} chunks from {len(set(c['module'] for c in chunks))} modules")

        return formatted_prompt

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "model_name": self.model_name,
            "collection_name": self.collection_name,
            "database_path": str(self.db_path),
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }


def main():
    """Main function for the RAG query system."""
    parser = argparse.ArgumentParser(
        description='RIOT OS Documentation RAG Query System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python riot_query.py "How do I configure GPIO pins?"
  python riot_query.py "RIOT networking stack" --module sys --chunks 10
  python riot_query.py "Timer configuration" --output prompt.txt
  python riot_query.py "Thread synchronization" --category "core" --prefer-module "core"
        """
    )

    parser.add_argument('query', help='Your question about RIOT OS')
    parser.add_argument('--db-path', default="./riot_vector_db",
                        help='Path to ChromaDB database')
    parser.add_argument('--collection', default="riot_docs",
                        help='Collection name in ChromaDB')
    parser.add_argument('--model', default="sentence-transformers/all-mpnet-base-v2",
                        help='Embedding model name')
    parser.add_argument('--chunks', type=int, default=8,
                        help='Number of chunks to retrieve (default: 8)')
    parser.add_argument('--module',
                        help='Prefer results from specific module')
    parser.add_argument('--category',
                        help='Prefer results from specific category')
    parser.add_argument('--output', '-o',
                        help='Save prompt to file')
    parser.add_argument('--stats', action='store_true',
                        help='Show system statistics')
    parser.add_argument('--print-chunks', action='store_true',
                        help='Print chunk details before prompt')

    args = parser.parse_args()

    try:
        # Initialize RAG system
        rag = RIOTRAGQuerySystem(
            db_path=args.db_path,
            collection_name=args.collection,
            model_name=args.model
        )

        # Show stats if requested
        if args.stats:
            stats = rag.get_system_stats()
            print("System Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()

        # Process query
        if args.print_chunks:
            # Get chunks first to show details
            chunks = rag.get_top_chunks(
                args.query,
                n_chunks=args.chunks,
                prefer_module=args.module,
                prefer_category=args.category
            )

            print(f"Retrieved {len(chunks)} chunks for query: '{args.query}'")
            print("\nChunk Details:")
            for chunk in chunks:
                print(f"  {chunk['rank']}. {chunk['module']}/{chunk['file_name']} "
                      f"(score: {chunk['relevance_score']:.3f})")
            print()

        # Generate the complete prompt
        prompt = rag.generate_query_response(
            args.query,
            n_chunks=args.chunks,
            prefer_module=args.module,
            prefer_category=args.category,
            output_file=args.output
        )

        # Output the prompt
        if not args.output:
            print(prompt)
        else:
            print(f"Prompt saved to: {args.output}")
            print(f"Prompt length: {len(prompt)} characters")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())