#!/usr/bin/env python3
import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_FILE = Path("riot_examples_chunks.jsonl")
OUTPUT_FILE = Path("riot_examples_embeddings.jsonl")

# Model options (uncomment the one you want to use):
# MODEL_NAME = "all-MiniLM-L6-v2"                    # Fast, good general purpose (384 dim)
MODEL_NAME = "all-mpnet-base-v2"                   # Better quality, slower (768 dim)
#MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"  # Good balance (384 dim)
# MODEL_NAME = "microsoft/codebert-base"             # Code-specific (768 dim)
# MODEL_NAME = "BAAI/bge-small-en-v1.5"             # Good performance, small (384 dim)

# Batch size for processing (adjust based on your RAM)
BATCH_SIZE = 32

def load_chunks(input_file):
    """Load chunks from JSONL file"""
    chunks = []
    print(f"Loading chunks from {input_file}...")
    
    if not input_file.exists():
        print(f"ERROR: Input file does not exist: {input_file}")
        return []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def embed_chunks(chunks, model, batch_size=32):
    """Generate embeddings for chunks in batches"""
    print(f"Generating embeddings using model: {model.get_sentence_embedding_dimension()}D {MODEL_NAME}")
    
    all_texts = [chunk['text'] for chunk in chunks]
    all_embeddings = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding batches"):
        batch_texts = all_texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        all_embeddings.extend(batch_embeddings)
        
        # Small delay to prevent overheating on CPU
        time.sleep(0.01)
    
    return all_embeddings

def save_embeddings(chunks, embeddings, output_file):
    """Save chunks with their embeddings to JSONL file"""
    print(f"Saving embeddings to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk, embedding in zip(chunks, embeddings):
            # Create new record with embedding
            record = {
                "source_example": chunk["source_example"],
                "file_path": chunk["file_path"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": embedding.tolist(),  # Convert numpy array to list for JSON
                "embedding_dim": len(embedding),
                "model_name": MODEL_NAME
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Successfully saved {len(chunks)} embeddings")

def main():
    print("=== RIOT Examples Embedding Generator ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Load chunks
    chunks = load_chunks(INPUT_FILE)
    if not chunks:
        print("No chunks to process. Exiting.")
        return
    
    # Load model
    print("Loading embedding model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have sentence-transformers installed:")
        print("pip install sentence-transformers")
        return
    
    # Generate embeddings
    embeddings = embed_chunks(chunks, model, BATCH_SIZE)
    
    # Save results
    save_embeddings(chunks, embeddings, OUTPUT_FILE)
    
    print("\n=== SUMMARY ===")
    print(f"Processed {len(chunks)} chunks")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Average text length: {np.mean([len(chunk['text']) for chunk in chunks]):.1f} characters")

if __name__ == "__main__":
    main()