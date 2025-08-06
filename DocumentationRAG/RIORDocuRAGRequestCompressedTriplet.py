#!/usr/bin/env python3
"""
RIOT OS Documentation Triplet Loss Compressed Embedding Search System

This script searches through compressed RIOT OS documentation chunks using
the triplet loss compressed embeddings and trained triplet autoencoder model.

Dependencies:
    pip install torch numpy sentence-transformers scikit-learn

Usage:
    python riot_triplet_search.py "How do I configure GPIO pins in RIOT?"
    python riot_triplet_search.py "RIOT networking stack usage" --module sys
    python riot_triplet_search.py "Timer configuration" --output prompt.txt

Author: Assistant
License: MIT
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pickle

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletEmbeddingAutoencoder(nn.Module):
    """Autoencoder with triplet loss supervision for semantic embedding compression."""

    def __init__(self, input_dim: int = 768, compressed_dim: int = 256):
        super(TripletEmbeddingAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.compressed_dim = compressed_dim

        # Encoder layers with progressive dimensionality reduction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, compressed_dim),
            nn.Tanh()  # Normalize to [-1, 1] range
        )

        # Decoder layers - mirror of encoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, input_dim),
            nn.Tanh()  # Match encoder output range
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x):
        """Encode input to compressed representation."""
        return self.encoder(x)

    def decode(self, x):
        """Decode compressed representation back to original dimension."""
        return self.decoder(x)

    def forward(self, x):
        """Full autoencoder forward pass."""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded


class RIOTTripletSearchSystem:
    """Search system for triplet loss compressed RIOT OS documentation embeddings."""

    def __init__(self, chunks_file: str,
                 model_file: str = "triplet_autoencoder_model.pth",
                 scaler_file: str = "triplet_embedding_scaler.pkl",
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):

        self.chunks_file = Path(chunks_file)
        self.model_file = Path(model_file)
        self.scaler_file = Path(scaler_file)
        self.embedding_model_name = embedding_model_name

        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = self._load_embedding_model()
        self.triplet_model = self._load_triplet_model()
        self.scaler = self._load_scaler()
        self.chunks = self._load_chunks()

        # Prepare embeddings matrix for efficient search
        self.embeddings_matrix = self._prepare_embeddings_matrix()

        logger.info(f"Triplet search system initialized with {len(self.chunks)} chunks")
        logger.info(f"Compressed embedding dimension: {self.embeddings_matrix.shape[1]}")

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(self.embedding_model_name, device=device)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _load_triplet_model(self) -> TripletEmbeddingAutoencoder:
        """Load the trained triplet autoencoder model."""
        if not self.model_file.exists():
            raise FileNotFoundError(f"Triplet model file not found: {self.model_file}")

        # Load model checkpoint
        checkpoint = torch.load(self.model_file, map_location=self.device)

        # Create model with saved dimensions
        model = TripletEmbeddingAutoencoder(
            input_dim=checkpoint['input_dim'],
            compressed_dim=checkpoint['compressed_dim']
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        logger.info(f"Loaded Triplet Autoencoder: {checkpoint['input_dim']} -> {checkpoint['compressed_dim']}")
        return model

    def _load_scaler(self) -> StandardScaler:
        """Load the embedding scaler."""
        if not self.scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_file}")

        with open(self.scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        logger.info("Loaded embedding scaler")
        return scaler

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        if not self.chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks from {self.chunks_file}")
        return chunks

    def _prepare_embeddings_matrix(self) -> np.ndarray:
        """Prepare the embeddings matrix for efficient similarity search."""
        embeddings = []
        for chunk in self.chunks:
            # Try both possible keys for triplet embeddings
            embedding_key = 'triplet_compressed_embedding'
            if embedding_key not in chunk:
                # Fallback to generic compressed embedding
                embedding_key = 'compressed_embedding'

            if embedding_key not in chunk:
                raise KeyError(f"No triplet compressed embedding found in chunk. Available keys: {list(chunk.keys())}")

            embedding = np.array(chunk[embedding_key], dtype=np.float32)
            embeddings.append(embedding)

        embeddings_matrix = np.vstack(embeddings)
        logger.info(f"Prepared triplet embeddings matrix: {embeddings_matrix.shape}")
        return embeddings_matrix

    def _enhance_query(self, query: str) -> str:
        """Enhance query with RIOT-specific context."""
        return f"RIOT OS documentation query: {query}"

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query using the embedding model and compress with triplet autoencoder."""
        # Enhance query
        enhanced_query = self._enhance_query(query)

        # Generate embedding using sentence transformer
        query_embedding = self.embedding_model.encode(
            [enhanced_query],
            normalize_embeddings=True
        )[0]  # Get first (and only) embedding

        # Normalize using the scaler
        normalized_embedding = self.scaler.transform(query_embedding.reshape(1, -1))

        # Compress using triplet autoencoder
        with torch.no_grad():
            query_tensor = torch.FloatTensor(normalized_embedding).to(self.device)
            compressed_embedding = self.triplet_model.encode(query_tensor)
            compressed_embedding = compressed_embedding.cpu().numpy().flatten()

        return compressed_embedding

    def _calculate_diversity_score(self, results_indices: List[int]) -> List[float]:
        """Calculate diversity scores to promote varied results."""
        diversity_scores = []
        seen_modules = set()
        seen_files = set()
        seen_categories = set()

        for idx in results_indices:
            chunk = self.chunks[idx]
            metadata = chunk['metadata']
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

    def _calculate_preference_scores(self, results_indices: List[int],
                                     query: str,
                                     prefer_module: Optional[str] = None,
                                     prefer_category: Optional[str] = None) -> List[float]:
        """Calculate preference scores based on query and preferences."""
        query_words = [w for w in query.lower().split() if w and len(w) > 3]
        preference_scores = []

        for idx in results_indices:
            chunk = self.chunks[idx]
            metadata = chunk['metadata']
            score = 0.0

            # Module preference
            if prefer_module and metadata.get('module', '').lower() == prefer_module.lower():
                score += 0.4

            # Category preference
            if prefer_category and metadata.get('category', '').lower() == prefer_category.lower():
                score += 0.3

            # Boost for API documentation
            if metadata.get('has_functions') == 'true':
                score += 0.1

            # Boost for example code
            if 'example' in metadata.get('file_path', '').lower():
                score += 0.2

            # Query word matches in file path, module, and category
            fp = metadata.get('file_path', '').lower()
            mod = metadata.get('module', '').lower()
            cat = metadata.get('category', '').lower()

            for w in query_words:
                if w in fp or w == mod or w == cat:
                    score += 0.1

            # Additional boost if query words are in file name
            file_name = metadata.get('file_name', '').lower()
            for w in query_words:
                if w in file_name:
                    score += 0.15

            # Special boost for triplet loss - semantic clustering should be better
            # Give extra points if content seems highly relevant
            content = chunk.get('content', '').lower()
            content_matches = sum(1 for w in query_words if w in content)
            if content_matches >= 2:
                score += 0.1 * content_matches

            preference_scores.append(score)

        return preference_scores

    def search_chunks(self, query: str,
                      n_results: int = 16,
                      prefer_module: Optional[str] = None,
                      prefer_category: Optional[str] = None) -> List[Tuple[int, float]]:
        """Search for relevant chunks using triplet loss compressed embeddings."""

        # Encode and compress query
        query_embedding = self._encode_query(query)

        # Calculate cosine similarity with all embeddings
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings_matrix
        )[0]

        # Get top results based on similarity
        top_indices = np.argsort(similarities)[::-1][:n_results]
        top_similarities = similarities[top_indices]

        # Calculate diversity and preference scores
        diversity_scores = self._calculate_diversity_score(top_indices.tolist())
        preference_scores = self._calculate_preference_scores(
            top_indices.tolist(), query, prefer_module, prefer_category
        )

        # Combine scores with weights (slightly favor similarity for triplet loss)
        combined_scores = []
        for i, (sim_score, div_score, pref_score) in enumerate(
                zip(top_similarities, diversity_scores, preference_scores)
        ):
            combined_score = (
                    0.65 * sim_score +  # Primary: semantic similarity (higher weight for triplet)
                    0.15 * div_score +  # Secondary: diversity
                    0.2 * pref_score  # Tertiary: preferences
            )
            combined_scores.append(combined_score)

        # Create result tuples and sort by combined score
        results = list(zip(top_indices, combined_scores))
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(results)} relevant chunks for query: {query}")
        return results

    def get_top_chunks(self, query: str,
                       n_chunks: int = 8,
                       prefer_module: Optional[str] = None,
                       prefer_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the top N most relevant chunks."""

        # Search with larger initial set for better reranking
        search_results = self.search_chunks(
            query,
            n_results=min(n_chunks * 2, 32),
            prefer_module=prefer_module,
            prefer_category=prefer_category
        )

        # Select top N chunks
        top_chunks = []
        for rank, (chunk_idx, score) in enumerate(search_results[:n_chunks]):
            chunk = self.chunks[chunk_idx].copy()  # Create a copy

            # Update chunk information
            chunk['rank'] = rank + 1
            chunk['relevance_score'] = float(score)

            # Ensure all required fields are present
            if 'file_name' not in chunk:
                chunk['file_name'] = Path(chunk.get('file_path', '')).name

            top_chunks.append(chunk)

        return top_chunks

    def format_claude_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Format the retrieved chunks into an optimized prompt for Claude."""

        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build the prompt
        prompt = f"""You are a RIOT OS expert assistant. Use the following documentation chunks to answer the user's question about RIOT OS.

**User Question:** {query}

**RIOT OS Documentation Context:**
The following {len(chunks)} chunks were retrieved from the official RIOT OS documentation using triplet loss compressed embeddings based on semantic similarity and relevance to your question:

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
        modules = set(c['module'] for c in chunks)
        logger.info(f"Retrieved {len(chunks)} chunks from {len(modules)} modules")

        return formatted_prompt

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_chunks": len(self.chunks),
            "embedding_model": self.embedding_model_name,
            "compressed_embedding_dimension": self.embeddings_matrix.shape[1],
            "original_embedding_dimension": self.triplet_model.input_dim,
            "compression_ratio": f"{self.triplet_model.input_dim}:{self.triplet_model.compressed_dim}",
            "model_type": "Triplet Loss Autoencoder",
            "chunks_file": str(self.chunks_file),
            "model_file": str(self.model_file),
            "device": str(self.device)
        }


def main():
    """Main function for the triplet loss compressed embedding search system."""
    parser = argparse.ArgumentParser(
        description='RIOT OS Documentation Triplet Loss Compressed Embedding Search System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python riot_triplet_search.py "How do I configure GPIO pins?"
  python riot_triplet_search.py "RIOT networking stack" --module sys --chunks 10
  python riot_triplet_search.py "Timer configuration" --output prompt.txt
  python riot_triplet_search.py "Thread synchronization" --category "core"
        """
    )

    parser.add_argument('query', help='Your question about RIOT OS')
    parser.add_argument('--chunks-file', default="triplet_compressed_chunks.json",
                        help='Path to triplet compressed chunks JSON file')
    parser.add_argument('--model-file', default="triplet_autoencoder_model.pth",
                        help='Path to trained triplet autoencoder model')
    parser.add_argument('--scaler-file', default="triplet_embedding_scaler.pkl",
                        help='Path to embedding scaler')
    parser.add_argument('--embedding-model', default="sentence-transformers/all-mpnet-base-v2",
                        help='Sentence transformer model name')
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
        # Initialize search system
        search_system = RIOTTripletSearchSystem(
            chunks_file=args.chunks_file,
            model_file=args.model_file,
            scaler_file=args.scaler_file,
            embedding_model_name=args.embedding_model
        )

        # Show stats if requested
        if args.stats:
            stats = search_system.get_system_stats()
            print("Triplet Loss Search System Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()

        # Process query
        if args.print_chunks:
            # Get chunks first to show details
            chunks = search_system.get_top_chunks(
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
        prompt = search_system.generate_query_response(
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