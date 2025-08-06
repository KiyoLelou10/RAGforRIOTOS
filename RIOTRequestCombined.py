#!/usr/bin/env python3
"""
RIOT OS Unified Documentation & Examples RAG Query System

This script retrieves the most relevant RIOT OS documentation chunks (primary) and
example code chunks (supplementary) for a user query, then formats them into an
optimized prompt for Claude or other LLM models.

The system prioritizes documentation chunks as the authoritative source and includes
example chunks as additional context. Some retrieved chunks may be irrelevant due to
the nature of semantic search.

Dependencies:
    pip install chromadb sentence-transformers torch numpy scikit-learn

Usage:
    python riot_unified_rag.py "How do I configure GPIO pins in RIOT?"
    python riot_unified_rag.py "RIOT networking stack usage" --doc-chunks 6 --example-chunks 4
    python riot_unified_rag.py "Timer configuration" --output prompt.txt

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
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RIOTUnifiedRAGSystem:
    """
    Unified RAG query system for RIOT OS documentation and examples.

    Prioritizes documentation chunks as primary sources and includes example chunks
    as supplementary context. The system may occasionally retrieve irrelevant chunks
    due to the probabilistic nature of semantic search.
    """

    def __init__(self,
                 doc_db_path: str = "./riot_vector_db",
                 doc_collection_name: str = "riot_docs",
                 examples_embeddings_file: str = "./riot_examples_embeddings.jsonl",
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the unified RAG system."""
        self.doc_db_path = Path(doc_db_path)
        self.doc_collection_name = doc_collection_name
        self.examples_embeddings_file = Path(examples_embeddings_file)
        self.model_name = model_name

        # Initialize components
        self.model = self._load_embedding_model()

        # Documentation system (ChromaDB)
        self.chroma_client = self._initialize_chromadb()
        self.doc_collection = self._load_doc_collection()

        # Examples system (JSONL embeddings)
        self.example_chunks = []
        self.example_embeddings_matrix = None
        self._load_example_embeddings()

        logger.info("Unified RAG system initialized successfully")

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence transformer model (consistent across both systems)."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded embedding model: {self.model_name} on {device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback model
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _initialize_chromadb(self) -> Optional[chromadb.Client]:
        """Initialize ChromaDB client for documentation."""
        if not self.doc_db_path.exists():
            logger.warning(f"Documentation database not found: {self.doc_db_path}")
            return None

        try:
            return chromadb.PersistentClient(path=str(self.doc_db_path))
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return None

    def _load_doc_collection(self) -> Optional[chromadb.Collection]:
        """Load the documentation collection from ChromaDB."""
        if not self.chroma_client:
            return None

        try:
            collection = self.chroma_client.get_collection(self.doc_collection_name)
            count = collection.count()
            logger.info(f"Loaded documentation collection with {count} chunks")
            return collection
        except Exception as e:
            logger.warning(f"Failed to load documentation collection: {e}")
            return None

    def _load_example_embeddings(self):
        """Load example chunks and embeddings from JSONL file."""
        if not self.examples_embeddings_file.exists():
            logger.warning(f"Examples embeddings file not found: {self.examples_embeddings_file}")
            return

        chunks = []
        embeddings = []

        try:
            with open(self.examples_embeddings_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        chunks.append(record)
                        embeddings.append(np.array(record['embedding']))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping invalid record on line {line_num}: {e}")
                        continue

            if embeddings:
                self.example_chunks = chunks
                self.example_embeddings_matrix = np.vstack(embeddings)
                logger.info(f"Loaded {len(self.example_chunks)} example chunks with embeddings")
            else:
                logger.warning("No valid example embeddings found")

        except Exception as e:
            logger.error(f"Failed to load example embeddings: {e}")

    def _enhance_query(self, query: str) -> str:
        """Enhance query with RIOT-specific context."""
        return f"RIOT OS documentation query: {query}"

    def _calculate_diversity_score(self, results: Dict[str, Any]) -> List[float]:
        """Calculate diversity scores for documentation results to promote variety."""
        if not results.get('metadatas') or not results['metadatas'][0]:
            return [0.0] * len(results.get('documents', [[]])[0])

        metadatas = results['metadatas'][0]
        diversity_scores = []
        seen_modules = set()
        seen_files = set()
        seen_categories = set()

        for metadata in metadatas:
            score = 0.0

            # Reward new modules, files, and categories
            module = metadata.get('module', '')
            if module and module not in seen_modules:
                score += 0.3
                seen_modules.add(module)

            file_name = metadata.get('file_name', '')
            if file_name and file_name not in seen_files:
                score += 0.2
                seen_files.add(file_name)

            category = metadata.get('category', '')
            if category and category not in seen_categories:
                score += 0.1
                seen_categories.add(category)

            diversity_scores.append(score)

        return diversity_scores

    def _rerank_doc_results(self, results: Dict[str, Any], query: str,
                            prefer_module: Optional[str] = None,
                            prefer_category: Optional[str] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        """Rerank documentation results using hybrid scoring."""
        if not results.get('documents') or not results['documents'][0]:
            return []

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Calculate component scores
        similarity_scores = [1 - d for d in distances]
        diversity_scores = self._calculate_diversity_score(results)
        query_words = [w.lower() for w in query.split() if len(w) > 3]

        # Calculate preference scores
        preference_scores = []
        for metadata in metadatas:
            score = 0.0

            # Module and category preferences
            if prefer_module and metadata.get('module', '').lower() == prefer_module.lower():
                score += 0.4
            if prefer_category and metadata.get('category', '').lower() == prefer_category.lower():
                score += 0.3

            # Content type boosts
            if metadata.get('has_functions') == 'true':
                score += 0.1
            if 'example' in metadata.get('file_path', '').lower():
                score += 0.2

            # Query keyword matches
            fp = metadata.get('file_path', '').lower()
            mod = metadata.get('module', '').lower()
            cat = metadata.get('category', '').lower()

            for word in query_words:
                if word in fp or word == mod or word == cat:
                    score += 0.1

            preference_scores.append(score)

        # Combine scores with weights
        combined_scores = []
        for i in range(len(documents)):
            combined_score = (
                    0.6 * similarity_scores[i] +  # Primary: semantic similarity
                    0.2 * diversity_scores[i] +  # Secondary: diversity
                    0.2 * preference_scores[i]  # Tertiary: preferences
            )
            combined_scores.append(combined_score)

        # Create and sort ranked results
        ranked_results = [(documents[i], metadatas[i], combined_scores[i])
                          for i in range(len(documents))]
        ranked_results.sort(key=lambda x: x[2], reverse=True)

        return ranked_results

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query for path matching."""
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())

        # Basic stopwords for technical content
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'how', 'what', 'where', 'when', 'why', 'use', 'using', 'configure'
        }

        return [word for word in words if word not in stopwords and len(word) > 1]

    def _calculate_path_boost(self, query: str, file_path: str, boost_value: float = 0.1) -> Tuple[float, List[str]]:
        """Calculate boost score for examples based on query keywords in file path."""
        keywords = self._extract_query_keywords(query)
        file_path_lower = file_path.lower()

        boost_score = 0.0
        matched_keywords = []

        for keyword in keywords:
            if keyword in file_path_lower:
                boost_score += boost_value
                matched_keywords.append(keyword)

        return boost_score, matched_keywords

    def search_documentation(self, query: str, n_results: int = 16,
                             prefer_module: Optional[str] = None,
                             prefer_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search documentation chunks with advanced reranking."""
        if not self.doc_collection:
            logger.warning("Documentation collection not available")
            return []

        try:
            # Enhance query and generate embedding
            enhanced_query = self._enhance_query(query)
            query_embedding = self.model.encode([enhanced_query], normalize_embeddings=True)

            # Search documentation
            search_kwargs = {
                "query_embeddings": query_embedding.tolist(),
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }

            results = self.doc_collection.query(**search_kwargs)

            # Rerank and format results
            ranked_results = self._rerank_doc_results(
                results, query, prefer_module, prefer_category
            )

            doc_chunks = []
            for i, (document, metadata, score) in enumerate(ranked_results):
                chunk_info = {
                    'rank': i + 1,
                    'content': document,
                    'metadata': metadata,
                    'relevance_score': score,
                    'source_type': 'documentation',
                    'file_path': metadata.get('file_path', ''),
                    'module': metadata.get('module', ''),
                    'category': metadata.get('category', ''),
                    'file_name': metadata.get('file_name', ''),
                    'section': metadata.get('section', ''),
                }
                doc_chunks.append(chunk_info)

            logger.info(f"Retrieved {len(doc_chunks)} documentation chunks")
            return doc_chunks

        except Exception as e:
            logger.error(f"Failed to search documentation: {e}")
            return []

    def search_examples(self, query: str, n_results: int = 8) -> List[Dict[str, Any]]:
        """Search example chunks using cosine similarity with path boosting."""
        if not self.example_chunks or self.example_embeddings_matrix is None:
            logger.warning("Example embeddings not available")
            return []

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], normalize_embeddings=True)

            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.example_embeddings_matrix)[0]

            # Calculate scores with path boosting
            results = []
            for i, chunk in enumerate(self.example_chunks):
                cosine_score = similarities[i]
                path_boost, matched_keywords = self._calculate_path_boost(query, chunk['file_path'])
                final_score = cosine_score + path_boost

                results.append({
                    'chunk': chunk,
                    'cosine_similarity': float(cosine_score),
                    'path_boost': float(path_boost),
                    'final_score': float(final_score),
                    'matched_keywords': matched_keywords
                })

            # Sort and select top results
            results.sort(key=lambda x: x['final_score'], reverse=True)
            top_results = results[:n_results]

            # Format as standardized chunks
            example_chunks = []
            for i, result in enumerate(top_results):
                chunk_data = result['chunk']
                chunk_info = {
                    'rank': i + 1,
                    'content': chunk_data['text'],
                    'relevance_score': result['final_score'],
                    'source_type': 'example',
                    'file_path': chunk_data['file_path'],
                    'chunk_id': chunk_data['chunk_id'],
                    'matched_keywords': result['matched_keywords'],
                    'cosine_similarity': result['cosine_similarity'],
                    'path_boost': result['path_boost']
                }
                example_chunks.append(chunk_info)

            logger.info(f"Retrieved {len(example_chunks)} example chunks")
            return example_chunks

        except Exception as e:
            logger.error(f"Failed to search examples: {e}")
            return []

    def format_claude_prompt(self, query: str, doc_chunks: List[Dict[str, Any]],
                             example_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into an optimized prompt for Claude."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt = f"""You are a RIOT OS expert assistant. Use the following documentation and example code to answer the user's question about RIOT OS.

**User Question:** {query}

**IMPORTANT CONTEXT HIERARCHY:**
1. **Primary Source**: Official RIOT OS documentation chunks (authoritative and comprehensive)
2. **Supplementary Source**: Example code chunks (practical implementations and usage patterns)

Note: Some retrieved chunks may be less relevant due to the nature of semantic search. Focus on the most pertinent information.

"""

        # Add documentation chunks (primary source)
        if doc_chunks:
            prompt += f"""
**PRIMARY: RIOT OS DOCUMENTATION** ({len(doc_chunks)} chunks)
These are the authoritative documentation sources that should guide your response:

"""
            for chunk in doc_chunks:
                prompt += f"""
---
**Doc Chunk {chunk['rank']}** (Relevance: {chunk['relevance_score']:.3f})
**Module:** {chunk.get('module', 'N/A')}
**Category:** {chunk.get('category', 'N/A')}
**File:** {chunk.get('file_name', 'N/A')}
**Path:** {chunk['file_path']}
**Section:** {chunk.get('section', 'N/A')}

```
{chunk['content']}
```

"""

        # Add example chunks (supplementary source)
        if example_chunks:
            prompt += f"""
**SUPPLEMENTARY: EXAMPLE CODE** ({len(example_chunks)} chunks)
These examples provide practical context but should supplement, not override, the documentation:

"""
            for chunk in example_chunks:
                prompt += f"""
---
**Example {chunk['rank']}** (Relevance: {chunk['relevance_score']:.3f}, Similarity: {chunk.get('cosine_similarity', 0):.3f})
**File:** {chunk['file_path']}
**Chunk ID:** {chunk.get('chunk_id', 'N/A')}
"""
                if chunk.get('matched_keywords'):
                    prompt += f"**Keywords Matched:** {', '.join(chunk['matched_keywords'])}\n"

                prompt += f"""
```
{chunk['content']}
```

"""

        # Add comprehensive instructions
        prompt += """
---

**RESPONSE INSTRUCTIONS:**

**Priority Guidelines:**
1. **Prioritize documentation chunks** - These contain authoritative information
2. **Use examples as practical context** - They show real implementations but may be incomplete
3. **Cross-reference both sources** - Combine authoritative guidance with practical examples
4. **Flag irrelevant chunks** - Some retrieved content may not be directly relevant to the query

**Response Structure:**
1. **Direct Answer**: Start with a clear, concise answer to the user's question
2. **Documentation-Based Guidance**: Provide comprehensive information from the official docs
3. **Practical Examples**: Include relevant code examples and usage patterns when available
4. **Implementation Details**: Cover configuration, dependencies, and prerequisites
5. **Best Practices**: Mention recommended approaches from the documentation

**Technical Requirements:**
- Use official RIOT OS terminology and conventions
- Reference specific modules, APIs, and functions accurately
- Include file paths and module names when helpful
- Provide step-by-step instructions for implementation tasks
- Mention any version-specific considerations if apparent

**Quality Checks:**
- Verify information consistency between documentation and examples
- Note any discrepancies or outdated example code
- Highlight incomplete information and suggest where to find more details
- Structure the response with clear headers and bullet points

**Final Notes:**
- If the documentation chunks don't fully address the query, clearly state what information is missing
- When examples contradict documentation, prioritize the documentation and note the discrepancy
- Include relevant warnings, limitations, or common pitfalls mentioned in the sources

Please provide a comprehensive, well-structured response based on the RIOT OS sources provided above.
"""

        return prompt

    def generate_unified_response(self, query: str,
                                  doc_chunks: int = 8,
                                  example_chunks: int = 8,
                                  prefer_module: Optional[str] = None,
                                  prefer_category: Optional[str] = None,
                                  output_file: Optional[str] = None) -> str:
        """Generate a complete unified RAG response."""
        logger.info(f"Processing unified query: {query}")

        # Search both sources
        doc_results = self.search_documentation(
            query,
            n_results=min(doc_chunks * 2, 32),  # Get more for better reranking
            prefer_module=prefer_module,
            prefer_category=prefer_category
        )[:doc_chunks]  # Take top N after reranking

        example_results = self.search_examples(query, n_results=example_chunks)

        # Generate availability summary
        doc_available = len(doc_results) > 0
        examples_available = len(example_results) > 0

        if not doc_available and not examples_available:
            return f"No relevant RIOT OS content found for query: {query}"

        # Format the unified prompt
        formatted_prompt = self.format_claude_prompt(query, doc_results, example_results)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_prompt)
            logger.info(f"Unified prompt saved to: {output_file}")

        # Log retrieval summary
        doc_modules = len(set(c.get('module', 'unknown') for c in doc_results)) if doc_results else 0
        logger.info(f"Retrieved {len(doc_results)} documentation chunks from {doc_modules} modules")
        logger.info(f"Retrieved {len(example_results)} example chunks")

        return formatted_prompt

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "documentation_available": self.doc_collection is not None,
            "examples_available": self.example_embeddings_matrix is not None
        }

        if self.doc_collection:
            stats.update({
                "doc_database_path": str(self.doc_db_path),
                "doc_collection_name": self.doc_collection_name,
                "total_doc_chunks": self.doc_collection.count()
            })

        if self.example_embeddings_matrix is not None:
            stats.update({
                "examples_file_path": str(self.examples_embeddings_file),
                "total_example_chunks": len(self.example_chunks)
            })

        return stats


def main():
    """Main function for the unified RAG query system."""
    parser = argparse.ArgumentParser(
        description='RIOT OS Unified Documentation & Examples RAG Query System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python riot_unified_rag.py "How do I configure GPIO pins?"
  python riot_unified_rag.py "RIOT networking stack" --doc-chunks 6 --example-chunks 4
  python riot_unified_rag.py "Timer configuration" --output prompt.txt --prefer-module core
  python riot_unified_rag.py "Thread synchronization" --stats
        """
    )

    # Required arguments
    parser.add_argument('query', help='Your question about RIOT OS')

    # Database configuration
    parser.add_argument('--doc-db-path', default="./riot_vector_db",
                        help='Path to documentation ChromaDB database')
    parser.add_argument('--doc-collection', default="riot_docs",
                        help='Documentation collection name in ChromaDB')
    parser.add_argument('--examples-file', default="./riot_examples_embeddings.jsonl",
                        help='Path to examples embeddings JSONL file')

    # Model configuration
    parser.add_argument('--model', default="sentence-transformers/all-mpnet-base-v2",
                        help='Embedding model name')

    # Retrieval configuration
    parser.add_argument('--doc-chunks', type=int, default=8,
                        help='Number of documentation chunks to retrieve (default: 8)')
    parser.add_argument('--example-chunks', type=int, default=8,
                        help='Number of example chunks to retrieve (default: 8)')

    # Search preferences
    parser.add_argument('--prefer-module',
                        help='Prefer results from specific module')
    parser.add_argument('--prefer-category',
                        help='Prefer results from specific category')

    # Output options
    parser.add_argument('--output', '-o',
                        help='Save prompt to file')
    parser.add_argument('--stats', action='store_true',
                        help='Show system statistics')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed chunk information')

    args = parser.parse_args()

    try:
        # Initialize unified RAG system
        rag = RIOTUnifiedRAGSystem(
            doc_db_path=args.doc_db_path,
            doc_collection_name=args.doc_collection,
            examples_embeddings_file=args.examples_file,
            model_name=args.model
        )

        # Show stats if requested
        if args.stats:
            stats = rag.get_system_stats()
            print("System Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()

        # Show detailed chunk info if verbose
        if args.verbose:
            doc_results = rag.search_documentation(
                args.query,
                n_results=args.doc_chunks,
                prefer_module=args.prefer_module,
                prefer_category=args.prefer_category
            )

            example_results = rag.search_examples(args.query, n_results=args.example_chunks)

            print(f"\nDetailed Results for: '{args.query}'")
            print(f"Documentation chunks: {len(doc_results)}")
            for chunk in doc_results:
                print(f"  {chunk['rank']}. {chunk.get('module', 'N/A')}/{chunk.get('file_name', 'N/A')} "
                      f"(score: {chunk['relevance_score']:.3f})")

            print(f"Example chunks: {len(example_results)}")
            for chunk in example_results:
                print(f"  {chunk['rank']}. {Path(chunk['file_path']).name} "
                      f"(score: {chunk['relevance_score']:.3f})")
            print()

        # Generate the unified prompt
        prompt = rag.generate_unified_response(
            args.query,
            doc_chunks=args.doc_chunks,
            example_chunks=args.example_chunks,
            prefer_module=args.prefer_module,
            prefer_category=args.prefer_category,
            output_file=args.output
        )

        # Output the prompt
        if not args.output:
            print(prompt)
        else:
            print(f"Unified prompt saved to: {args.output}")
            print(f"Prompt length: {len(prompt)} characters")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())