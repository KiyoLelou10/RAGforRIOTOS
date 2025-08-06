#!/usr/bin/env python3
"""
RIOT OS Documentation Embedder

This script takes the chunked RIOT OS documentation and creates embeddings
for each chunk, storing them in a local ChromaDB vector database for
efficient similarity search and retrieval with rich metadata support.

Dependencies:
    pip install chromadb sentence-transformers torch numpy tqdm

Author: Assistant
License: MIT
"""

import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import logging
import re

# Vector database and embedding imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"  # Best for technical docs
    batch_size: int = 32
    max_seq_length: int = 512  # Increased for technical documentation
    normalize_embeddings: bool = True
    device: str = "auto"  # "auto", "cpu", "cuda"
    include_metadata_in_embedding: bool = True  # Enhance embeddings with metadata


class RIOTEmbedder:
    """Embedder for RIOT OS documentation chunks with advanced metadata handling."""

    def __init__(self, config: EmbeddingConfig, db_path: str = "./riot_vector_db"):
        self.config = config
        self.db_path = Path(db_path)

        # Initialize embedding model
        self.model = self._load_embedding_model()

        # Initialize ChromaDB
        self.chroma_client = self._initialize_chromadb()
        self.collection = None

        logger.info(f"Initialized embedder with model: {config.model_name}")
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"Device: {self.model.device}")

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        try:
            # Determine device
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device

            logger.info(f"Loading embedding model: {self.config.model_name}")

            # Load model with specific configuration
            model = SentenceTransformer(
                self.config.model_name,
                device=device
            )

            # Set max sequence length
            model.max_seq_length = self.config.max_seq_length

            return model

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a reliable model
            logger.info("Falling back to all-MiniLM-L6-v2")
            return SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=device
            )

    def _initialize_chromadb(self) -> chromadb.Client:
        """Initialize ChromaDB client."""
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Use persistent client
        client = chromadb.PersistentClient(path=str(self.db_path))

        logger.info("ChromaDB initialized successfully")
        return client

    def create_collection(self, collection_name: str = "riot_docs") -> None:
        """Create or get a collection in ChromaDB."""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            # Create new collection with metadata schema
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "RIOT OS documentation chunks with embeddings",
                    "embedding_model": self.config.model_name,
                    "version": "1.0"
                }
            )
            logger.info(f"Created new collection: {collection_name}")

    def load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        logger.info(f"Loading chunks from: {chunks_file}")

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks

    def _normalize_metadata_value(self, value: Any) -> str:
        """Normalize metadata values for ChromaDB storage."""
        if value is None:
            return ""
        elif isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value if v is not None)
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value).strip()

    def _extract_module_hierarchy(self, file_path: str) -> Dict[str, str]:
        """Extract module hierarchy from file path."""
        path_parts = Path(file_path).parts

        # Extract main module categories
        main_module = ""
        sub_module = ""
        api_type = ""

        # Look for common RIOT module patterns
        for i, part in enumerate(path_parts):
            if part in ["sys", "core", "drivers", "pkg", "cpu", "boards"]:
                main_module = part
                if i + 1 < len(path_parts):
                    sub_module = path_parts[i + 1]
                break
            elif "include" in part.lower():
                if i + 1 < len(path_parts):
                    api_type = "header"
                    if not main_module and i + 2 < len(path_parts):
                        main_module = path_parts[i + 1]
                        sub_module = path_parts[i + 2] if i + 2 < len(path_parts) else ""

        return {
            "main_module": main_module,
            "sub_module": sub_module,
            "api_type": api_type
        }

    def _extract_content_features(self, content: str) -> Dict[str, str]:
        """Extract content-based features for metadata."""
        features = {}

        # Detect content types
        if "#define" in content:
            features["has_defines"] = "true"
        if "struct" in content:
            features["has_structs"] = "true"
        if "typedef" in content:
            features["has_typedefs"] = "true"
        if "enum" in content:
            features["has_enums"] = "true"
        if re.search(r'\b\w+\s*\([^)]*\)\s*{', content):
            features["has_functions"] = "true"
        if "/**" in content or "/*!" in content:
            features["has_documentation"] = "true"
        if "#include" in content:
            features["has_includes"] = "true"

        # Detect API levels
        if "static inline" in content:
            features["api_level"] = "inline"
        elif "static" in content:
            features["api_level"] = "static"
        elif "extern" in content:
            features["api_level"] = "extern"
        else:
            features["api_level"] = "public"

        return features

    def prepare_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare metadata for ChromaDB storage with best practices."""
        processed_metadata = []

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')

            # Start with normalized base metadata
            processed_meta = {}

            # Core file information
            file_path = metadata.get('file_path', '')
            processed_meta['file_path'] = self._normalize_metadata_value(file_path)
            processed_meta['file_name'] = Path(file_path).name if file_path else ""
            processed_meta['file_extension'] = Path(file_path).suffix if file_path else ""

            # Module hierarchy
            hierarchy_info = self._extract_module_hierarchy(file_path)
            processed_meta.update(hierarchy_info)

            # Original metadata with normalization
            processed_meta['module'] = self._normalize_metadata_value(metadata.get('module', ''))
            processed_meta['submodule'] = self._normalize_metadata_value(metadata.get('submodule', ''))
            processed_meta['category'] = self._normalize_metadata_value(metadata.get('category', ''))
            processed_meta['title'] = self._normalize_metadata_value(metadata.get('title', ''))
            processed_meta['section'] = self._normalize_metadata_value(metadata.get('section', ''))
            processed_meta['chunk_id'] = self._normalize_metadata_value(metadata.get('chunk_id', ''))

            # Hierarchy as searchable string
            hierarchy = metadata.get('hierarchy', [])
            processed_meta['hierarchy'] = self._normalize_metadata_value(hierarchy)
            processed_meta['hierarchy_depth'] = str(len(hierarchy))

            # Content statistics
            processed_meta['content_length'] = str(len(content))
            processed_meta['token_count'] = str(metadata.get('token_count', 0))
            processed_meta['has_overlap'] = "true" if metadata.get('overlap_with') else "false"

            # Content features
            content_features = self._extract_content_features(content)
            processed_meta.update(content_features)

            # URL and path information
            url_path = metadata.get('url_path', '')
            processed_meta['url_path'] = self._normalize_metadata_value(url_path)
            processed_meta['is_html'] = "true" if url_path.endswith('.html') else "false"

            # Create searchable tags
            tags = []
            tags.append(processed_meta['module'])
            tags.append(processed_meta['category'])
            tags.append(processed_meta['main_module'])
            tags.append(processed_meta['sub_module'])
            tags.extend(content_features.keys())

            # Remove empty tags and create searchable string
            tags = [tag for tag in tags if tag and tag != ""]
            processed_meta['tags'] = ",".join(tags)

            # Add embedding enhancement flag
            processed_meta['embedding_enhanced'] = "true" if self.config.include_metadata_in_embedding else "false"

            processed_metadata.append(processed_meta)

        return processed_metadata

    def create_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """Create enhanced embedding text with metadata context."""
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})

        if not self.config.include_metadata_in_embedding:
            return content

        # Create contextual prefix for better embeddings
        context_parts = []

        # Module context
        if metadata.get('module'):
            context_parts.append(f"RIOT {metadata['module']} module")

        # Category context
        if metadata.get('category'):
            context_parts.append(f"{metadata['category']} component")

        # File context
        file_path = metadata.get('file_path', '')
        if file_path:
            file_name = Path(file_path).name
            context_parts.append(f"from {file_name}")

        # Section context
        if metadata.get('section') and metadata['section'] != 'Unknown':
            context_parts.append(f"in {metadata['section']} section")

        # Combine context with content
        if context_parts:
            context_prefix = " ".join(context_parts) + ": "
            # Ensure we don't exceed token limits
            max_content_length = (self.config.max_seq_length * 4) - len(context_prefix)
            truncated_content = content[:max_content_length] if len(content) > max_content_length else content
            return context_prefix + truncated_content
        else:
            return content

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for chunks with enhanced text."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Prepare embedding texts
        embedding_texts = [self.create_embedding_text(chunk) for chunk in chunks]

        # Generate embeddings in batches
        all_embeddings = []

        for i in tqdm(range(0, len(embedding_texts), self.config.batch_size),
                      desc="Generating embeddings"):
            batch_texts = embedding_texts[i:i + self.config.batch_size]

            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Create zero embeddings as fallback
                fallback_embeddings = np.zeros((len(batch_texts), self.model.get_sentence_embedding_dimension()))
                all_embeddings.append(fallback_embeddings)

        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Store embeddings and metadata in ChromaDB."""
        logger.info("Storing embeddings in ChromaDB")

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = self.prepare_metadata(chunks)

        for i, chunk in enumerate(chunks):
            # Generate consistent chunk ID
            chunk_id = chunk.get('metadata', {}).get('chunk_id')
            if not chunk_id:
                chunk_id = self._generate_chunk_id(chunk, i)

            ids.append(chunk_id)
            documents.append(chunk['content'])

        # Convert embeddings to list format
        embeddings_list = embeddings.tolist()

        # Store in ChromaDB in batches
        batch_size = 100  # ChromaDB batch size

        for i in tqdm(range(0, len(ids), batch_size), desc="Storing in database"):
            batch_end = min(i + batch_size, len(ids))

            try:
                self.collection.add(
                    ids=ids[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    embeddings=embeddings_list[i:batch_end]
                )
            except Exception as e:
                logger.error(f"Error storing batch {i}: {e}")
                continue

        logger.info(f"Successfully stored {len(ids)} embeddings")

    def _generate_chunk_id(self, chunk: Dict[str, Any], index: int) -> str:
        """Generate a unique chunk ID."""
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '')

        # Create a reproducible ID based on content and metadata
        id_components = [
            metadata.get('file_path', ''),
            metadata.get('module', ''),
            metadata.get('section', ''),
            str(index),
            content[:50]  # First 50 chars for uniqueness
        ]

        id_string = "|".join(str(comp) for comp in id_components)
        return hashlib.md5(id_string.encode()).hexdigest()[:16]

    def embed_chunks(self, chunks_file: str, collection_name: str = "riot_docs") -> None:
        """Complete embedding pipeline."""
        logger.info("Starting embedding pipeline")

        # Load chunks
        chunks = self.load_chunks(chunks_file)

        # Create collection
        self.create_collection(collection_name)

        # Check if collection already has data
        existing_count = self.collection.count()
        if existing_count > 0:
            logger.info(f"Collection already contains {existing_count} embeddings")
            response = input("Do you want to clear the existing data? (y/N): ")
            if response.lower() == 'y':
                # Delete and recreate collection
                self.chroma_client.delete_collection(collection_name)
                self.create_collection(collection_name)
                logger.info("Cleared existing data")
            else:
                logger.info("Skipping embedding generation")
                return

        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Store embeddings
        self.store_embeddings(chunks, embeddings)

        logger.info("Embedding pipeline completed successfully")

    def search_similar(self, query: str, n_results: int = 5,
                       where: Optional[Dict[str, Any]] = None,
                       where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search for similar chunks with metadata filtering."""
        if not self.collection:
            raise ValueError("No collection available. Run embed_chunks first.")

        # Generate query embedding with same enhancement as documents
        if self.config.include_metadata_in_embedding:
            enhanced_query = f"RIOT OS documentation query: {query}"
        else:
            enhanced_query = query

        query_embedding = self.model.encode(
            [enhanced_query],
            normalize_embeddings=self.config.normalize_embeddings
        )

        # Search in ChromaDB with optional metadata filtering
        search_kwargs = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        if where:
            search_kwargs["where"] = where
        if where_document:
            search_kwargs["where_document"] = where_document

        results = self.collection.query(**search_kwargs)
        return results

    def search_by_module(self, query: str, module: str, n_results: int = 5) -> Dict[str, Any]:
        """Search within a specific module."""
        return self.search_similar(
            query=query,
            n_results=n_results,
            where={"module": module}
        )

    def search_by_category(self, query: str, category: str, n_results: int = 5) -> Dict[str, Any]:
        """Search within a specific category."""
        return self.search_similar(
            query=query,
            n_results=n_results,
            where={"category": category}
        )

    def search_by_file_type(self, query: str, file_extension: str, n_results: int = 5) -> Dict[str, Any]:
        """Search within specific file types."""
        return self.search_similar(
            query=query,
            n_results=n_results,
            where={"file_extension": file_extension}
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the collection."""
        if not self.collection:
            return {"error": "No collection available"}

        count = self.collection.count()

        # Get sample of metadata to analyze
        if count > 0:
            sample_results = self.collection.query(
                query_embeddings=[[0.0] * self.model.get_sentence_embedding_dimension()],
                n_results=min(100, count),
                include=["metadatas"]
            )

            # Analyze metadata distribution
            modules = set()
            categories = set()
            file_types = set()

            for metadata in sample_results.get('metadatas', [{}])[0]:
                modules.add(metadata.get('module', ''))
                categories.add(metadata.get('category', ''))
                file_types.add(metadata.get('file_extension', ''))

            return {
                "total_chunks": count,
                "model_name": self.config.model_name,
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "collection_name": self.collection.name,
                "unique_modules": len(modules),
                "unique_categories": len(categories),
                "unique_file_types": len(file_types),
                "sample_modules": list(modules)[:10],
                "sample_categories": list(categories)[:10],
                "sample_file_types": list(file_types)[:10]
            }

        return {
            "total_chunks": count,
            "model_name": self.config.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "collection_name": self.collection.name if self.collection else None
        }


def main():
    """Main function to run the RIOT documentation embedder."""
    parser = argparse.ArgumentParser(description='RIOT OS Documentation Embedder')
    parser.add_argument('chunks_file', help='Path to JSON file with chunks')
    parser.add_argument('--model', default="sentence-transformers/all-mpnet-base-v2",
                        help='Embedding model to use')
    parser.add_argument('--db-path', default="./riot_vector_db",
                        help='Path to store ChromaDB database')
    parser.add_argument('--collection', default="riot_docs",
                        help='Collection name in ChromaDB')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding generation')
    parser.add_argument('--device', default="auto",
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--no-metadata-enhancement', action='store_true',
                        help='Disable metadata enhancement in embeddings')
    parser.add_argument('--test-query',
                        help='Test query to run after embedding')
    parser.add_argument('--test-module',
                        help='Test module-specific search')
    parser.add_argument('--stats', action='store_true',
                        help='Show collection statistics')

    args = parser.parse_args()

    # Validate chunks file
    if not Path(args.chunks_file).exists():
        logger.error(f"Chunks file not found: {args.chunks_file}")
        return 1

    # Create embedding configuration
    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        include_metadata_in_embedding=not args.no_metadata_enhancement
    )

    # Create embedder
    embedder = RIOTEmbedder(config, args.db_path)

    try:
        # Run embedding pipeline
        embedder.embed_chunks(args.chunks_file, args.collection)

        # Show statistics
        if args.stats:
            stats = embedder.get_collection_stats()
            print(f"\nCollection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        # Test general query
        if args.test_query:
            logger.info(f"Testing query: {args.test_query}")
            results = embedder.search_similar(args.test_query, n_results=3)

            print(f"\nTop 3 results for '{args.test_query}':")
            for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                print(f"\n{i + 1}. Score: {1 - distance:.3f}")
                print(f"   Module: {metadata.get('module', 'N/A')}")
                print(f"   Category: {metadata.get('category', 'N/A')}")
                print(f"   File: {metadata.get('file_name', 'N/A')}")
                print(f"   Tags: {metadata.get('tags', 'N/A')}")
                print(f"   Content: {doc[:200]}...")

        # Test module-specific search
        if args.test_module and args.test_query:
            logger.info(f"Testing module search: {args.test_query} in {args.test_module}")
            results = embedder.search_by_module(args.test_query, args.test_module, n_results=3)

            print(f"\nTop 3 results for '{args.test_query}' in module '{args.test_module}':")
            for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                print(f"\n{i + 1}. Score: {1 - distance:.3f}")
                print(f"   File: {metadata.get('file_name', 'N/A')}")
                print(f"   Content: {doc[:200]}...")

        logger.info("Embedding completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        return 1


if __name__ == '__main__':
    exit(main())