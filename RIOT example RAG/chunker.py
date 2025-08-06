#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === CONFIGURATION ===
# Use the correct WSL/Linux path
ROOT_DIR = Path("/mnt/c/Users/asums/OneDrive/Desktop/RIOT/examples")
OUT_FILE = Path("riot_examples_chunks.jsonl")
# ~800 tokens (~2000 chars), 200-token (~500 chars) overlap
CHUNK_SIZE   = 2000
CHUNK_OVERLAP= 500

# File types to ingest
MD_EXTS   = {".md", ".rst"}
CODE_EXTS = {".c", ".h", ".cpp", ".py", ".ino"}

# initialize splitter
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def chunk_markdown(text):
    """
    First split on top-level headings, then apply recursive splitter
    """
    # preserve the leading '#' when splitting
    parts = re.split(r'(?m)^(?=# )', text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for sub in splitter.split_text(part):
            chunks.append(sub)
    return chunks

def chunk_code(text):
    """
    Try to split at function definitions; else fallback to recursive splitter
    """
    # rudimentary split at C-style function headers (lines ending with '{')
    parts = re.split(r'(?m)^(?:[A-Za-z_][\w\:<>]+\s+)+[A-Za-z_][\w_]*\s*\(.*\)\s*\{', text)
    # if the split produced fewer than 2 parts, just do the recursive split:
    if len(parts) < 2:
        return splitter.split_text(text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for sub in splitter.split_text(part):
            chunks.append(sub)
    return chunks

def process_file(path: Path):
    """Process a single file and return chunks"""
    try:
        print(f"Processing: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        print(f"  File size: {len(text)} characters")
        
        if not text.strip():
            print(f"  Warning: File is empty or whitespace only")
            return []
        
        ext = path.suffix.lower()
        if ext in MD_EXTS:
            chunks = chunk_markdown(text)
        elif ext in CODE_EXTS:
            chunks = chunk_code(text)
        else:
            print(f"  Warning: Unsupported file extension: {ext}")
            return []
        
        print(f"  Generated {len(chunks)} chunks")
        return chunks
    
    except Exception as e:
        print(f"  Error processing {path}: {e}")
        return []

def main():
    print(f"Starting chunking process...")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Output file: {OUT_FILE}")
    print(f"Looking for extensions: {MD_EXTS.union(CODE_EXTS)}")
    
    # Check if root directory exists
    if not ROOT_DIR.exists():
        print(f"ERROR: Root directory does not exist: {ROOT_DIR}")
        return
    
    if not ROOT_DIR.is_dir():
        print(f"ERROR: Root path is not a directory: {ROOT_DIR}")
        return
    
    total_files_found = 0
    total_files_processed = 0
    total_chunks = 0
    
    with OUT_FILE.open("w", encoding="utf-8") as out_f:
        for root, dirs, files in os.walk(ROOT_DIR):
            print(f"\nScanning directory: {root}")
            print(f"  Found {len(files)} files")
            
            for fname in files:
                path = Path(root) / fname
                total_files_found += 1
                
                # Check file extension
                if path.suffix.lower() not in MD_EXTS.union(CODE_EXTS):
                    print(f"  Skipping (unsupported extension): {fname}")
                    continue
                
                total_files_processed += 1
                chunks = process_file(path)
                
                if not chunks:
                    print(f"  No chunks generated for: {fname}")
                    continue
                
                for idx, chunk in enumerate(chunks, start=1):
                    # Create a cleaner file path by removing the base part
                    clean_path = str(path).replace("/mnt/c/Users/asums/OneDrive/Desktop/", "")
                    
                    record = {
                        "source_example": str(path.relative_to(ROOT_DIR)),
                        "file_path":       clean_path,
                        "chunk_id":        f"{path.stem}_chunk{idx:04d}",
                        "text":            chunk
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Total files found: {total_files_found}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Output written to: {OUT_FILE}")
    
    if total_chunks == 0:
        print("\nNo chunks were generated. Possible issues:")
        print("1. No files with supported extensions found")
        print("2. All files are empty or contain only whitespace")
        print("3. Permission issues reading files")
        print("4. Directory structure different than expected")

if __name__ == "__main__":
    main()