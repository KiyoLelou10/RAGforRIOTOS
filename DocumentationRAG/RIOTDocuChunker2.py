#!/usr/bin/env python3
"""
RIOT OS Documentation Chunking System for RAG
==============================================

This script processes RIOT OS documentation to create overlapping chunks
suitable for Retrieval-Augmented Generation (RAG) systems.

Features:
- Hierarchical chunking based on documentation structure
- Overlapping chunks with configurable overlap
- Rich metadata extraction (module, path, hierarchy)
- Tokenization using tiktoken (OpenAI's tokenizer)
- JSON output with detailed chunk information
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import tiktoken
from urllib.parse import urlparse, unquote
import hashlib


@dataclass
class ChunkMetadata:
    """Metadata for each documentation chunk"""
    module: str  # e.g., "gnrc", "udp", "core"
    submodule: str  # e.g., "netif", "routing", "ipv6"
    category: str  # e.g., "net", "sys", "drivers", "core"
    file_path: str  # Original file path
    url_path: str  # URL path from documentation
    hierarchy: List[str]  # Hierarchical path like ["sys", "net", "gnrc", "netif"]
    title: str  # Page/section title
    section: str  # Current section name
    chunk_id: str  # Unique identifier for the chunk
    token_count: int  # Number of tokens in the chunk
    overlap_with: List[str]  # IDs of chunks this overlaps with


@dataclass
class DocumentChunk:
    """A chunk of documentation with metadata"""
    content: str
    tokens: List[str]
    metadata: ChunkMetadata


class RIOTDocumentationChunker:
    """
    Chunker for RIOT OS documentation with support for overlapping chunks
    and rich metadata extraction.
    """

    def __init__(self,
                 chunk_size: int = 800,  # Larger chunks for documentation
                 overlap_size: int = 150,  # Overlap between chunks
                 encoding_name: str = "cl100k_base"):  # OpenAI's tiktoken encoding
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for chunks in tokens
            overlap_size: Number of tokens to overlap between chunks
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.encoding = tiktoken.get_encoding(encoding_name)

        # RIOT-specific module patterns
        self.module_patterns = {
            'gnrc': r'gnrc|GNRC',
            'udp': r'udp|UDP',
            'tcp': r'tcp|TCP',
            'ipv6': r'ipv6|IPv6',
            'ipv4': r'ipv4|IPv4',
            'coap': r'coap|CoAP',
            'lwm2m': r'lwm2m|LWM2M',
            'netif': r'netif|network interface',
            'routing': r'routing|RPL',
            'mac': r'mac|MAC',
            'phy': r'phy|PHY',
            'crypto': r'crypto|aes|hash|cipher',
            'shell': r'shell|Shell',
            'xtimer': r'xtimer|timer',
            'mutex': r'mutex|Mutex',
            'thread': r'thread|Thread',
            'ipc': r'ipc|IPC|messaging',
            'periph': r'periph|peripheral',
            'gpio': r'gpio|GPIO',
            'spi': r'spi|SPI',
            'i2c': r'i2c|I2C',
            'uart': r'uart|UART',
            'adc': r'adc|ADC',
            'pwm': r'pwm|PWM',
            'can': r'can|CAN',
            'usb': r'usb|USB'
        }

        # Category mapping based on RIOT structure
        self.category_mapping = {
            'core': ['core', 'kernel', 'scheduler', 'thread', 'ipc', 'mutex'],
            'sys': ['sys', 'shell', 'xtimer', 'crypto', 'data structures', 'posix'],
            'net': ['net', 'gnrc', 'udp', 'tcp', 'ipv6', 'ipv4', 'coap', 'lwm2m', 'netif', 'routing'],
            'drivers': ['drivers', 'periph', 'gpio', 'spi', 'i2c', 'uart', 'adc', 'pwm', 'can', 'usb'],
            'cpu': ['cpu', 'cortex', 'arm', 'msp430', 'avr', 'esp'],
            'boards': ['boards', 'board', 'platform']
        }

    def _generate_chunk_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate unique chunk ID based on content and metadata"""
        combined = f"{metadata.get('file_path', '')}{content[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _extract_hierarchy_from_path(self, file_path: str) -> List[str]:
        """Extract hierarchical path from file path"""
        path_parts = Path(file_path).parts
        hierarchy = []

        # Remove common prefixes and file extensions
        for part in path_parts:
            if part.endswith('.html'):
                part = part[:-5]
            if part not in ['doc', 'api', 'group__']:
                hierarchy.append(part.replace('group__', '').replace('__', '_'))

        return hierarchy

    def _detect_module_and_category(self, content: str, file_path: str) -> Tuple[str, str, str]:
        """
        Detect module, submodule, and category from content and file path.

        Returns:
            Tuple of (module, submodule, category)
        """
        content_lower = content.lower()
        path_lower = file_path.lower()

        # Detect module
        detected_module = "unknown"
        detected_submodule = ""
        detected_category = "unknown"

        # Check file path first for more accurate detection
        for module, pattern in self.module_patterns.items():
            if re.search(pattern, path_lower, re.IGNORECASE):
                detected_module = module
                break

        # If not found in path, check content
        if detected_module == "unknown":
            for module, pattern in self.module_patterns.items():
                if re.search(pattern, content_lower, re.IGNORECASE):
                    detected_module = module
                    break

        # Extract submodule from path
        if 'gnrc' in path_lower:
            submodule_match = re.search(r'gnrc[/_]([a-zA-Z0-9_]+)', path_lower)
            if submodule_match:
                detected_submodule = submodule_match.group(1)

        # Determine category
        for category, modules in self.category_mapping.items():
            if any(mod in path_lower or mod in content_lower for mod in modules):
                detected_category = category
                break

        return detected_module, detected_submodule, detected_category

    def _extract_title_and_section(self, content: str) -> Tuple[str, str]:
        """Extract title and current section from content"""
        lines = content.split('\n')
        title = "Unknown"
        section = "Unknown"

        # Look for title in first few lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if line and not line.startswith('#'):
                title = line[:100]  # Limit title length
                break

        # Look for section headers
        for line in lines:
            if line.startswith('#') or line.startswith('**'):
                section = line.strip('#* ')[:100]
                break

        return title, section

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using tiktoken"""
        tokens = self.encoding.encode(text)
        return [self.encoding.decode([token]) for token in tokens]

    def _create_overlapping_chunks(self, tokens: List[str], file_path: str) -> List[DocumentChunk]:
        """Create overlapping chunks from tokenized text"""
        chunks = []
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            # Single chunk
            content = ''.join(tokens)
            module, submodule, category = self._detect_module_and_category(content, file_path)
            hierarchy = self._extract_hierarchy_from_path(file_path)
            title, section = self._extract_title_and_section(content)

            chunk_id = self._generate_chunk_id(content, {'file_path': file_path})

            metadata = ChunkMetadata(
                module=module,
                submodule=submodule,
                category=category,
                file_path=file_path,
                url_path=file_path,
                hierarchy=hierarchy,
                title=title,
                section=section,
                chunk_id=chunk_id,
                token_count=total_tokens,
                overlap_with=[]
            )

            chunks.append(DocumentChunk(
                content=content,
                tokens=tokens,
                metadata=metadata
            ))
        else:
            # Multiple overlapping chunks
            start_idx = 0
            chunk_ids = []

            while start_idx < total_tokens:
                end_idx = min(start_idx + self.chunk_size, total_tokens)
                chunk_tokens = tokens[start_idx:end_idx]
                content = ''.join(chunk_tokens)

                # Extract metadata
                module, submodule, category = self._detect_module_and_category(content, file_path)
                hierarchy = self._extract_hierarchy_from_path(file_path)
                title, section = self._extract_title_and_section(content)

                chunk_id = self._generate_chunk_id(content, {'file_path': file_path})
                chunk_ids.append(chunk_id)

                # Determine overlaps
                overlap_with = []
                if len(chunk_ids) > 1:
                    overlap_with.append(chunk_ids[-2])  # Previous chunk
                if end_idx < total_tokens:
                    # Will overlap with next chunk
                    pass

                metadata = ChunkMetadata(
                    module=module,
                    submodule=submodule,
                    category=category,
                    file_path=file_path,
                    url_path=file_path,
                    hierarchy=hierarchy,
                    title=title,
                    section=section,
                    chunk_id=chunk_id,
                    token_count=len(chunk_tokens),
                    overlap_with=overlap_with
                )

                chunks.append(DocumentChunk(
                    content=content,
                    tokens=chunk_tokens,
                    metadata=metadata
                ))

                # Move to next chunk with overlap
                if end_idx >= total_tokens:
                    break
                start_idx = end_idx - self.overlap_size

            # Update overlap information
            for i, chunk in enumerate(chunks):
                if i < len(chunks) - 1:
                    chunk.metadata.overlap_with.append(chunks[i + 1].metadata.chunk_id)

        return chunks

    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content to extract text"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """Process a single documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Clean HTML if necessary
            if file_path.endswith('.html'):
                content = self._clean_html_content(content)

            # Tokenize content
            tokens = self._tokenize_text(content)

            # Create overlapping chunks
            chunks = self._create_overlapping_chunks(tokens, file_path)

            return chunks

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def process_directory(self, doc_dir: str) -> List[DocumentChunk]:
        """Process entire documentation directory recursively"""
        all_chunks = []
        doc_path = Path(doc_dir)

        if not doc_path.exists():
            raise FileNotFoundError(f"Documentation directory not found: {doc_dir}")

        # Find all documentation files recursively
        file_patterns = ['*.html', '*.md', '*.rst', '*.txt']
        doc_files = []

        print(f"Scanning directory: {doc_path}")
        print("Directory contents:")
        for item in doc_path.iterdir():
            print(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")

        for pattern in file_patterns:
            found_files = list(doc_path.rglob(pattern))
            print(f"Found {len(found_files)} files matching {pattern}")
            doc_files.extend(found_files)

        # Sort files for consistent processing order
        doc_files.sort()

        print(f"\nFound {len(doc_files)} documentation files total")

        if not doc_files:
            print("No documentation files found! Checking for other file types...")
            # Check for any files in the directory
            all_files = list(doc_path.rglob('*'))
            file_types = set()
            for f in all_files:
                if f.is_file():
                    file_types.add(f.suffix.lower())
            print(f"File types found: {sorted(file_types)}")

            # Ask if user wants to process other file types
            if '.htm' in file_types or '.xhtml' in file_types:
                print("Found .htm or .xhtml files, adding to processing...")
                doc_files.extend(doc_path.rglob('*.htm'))
                doc_files.extend(doc_path.rglob('*.xhtml'))

        processed_count = 0
        for file_path in doc_files:
            try:
                print(f"Processing [{processed_count + 1}/{len(doc_files)}]: {file_path}")
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        print(f"\nSuccessfully processed {processed_count} files")
        print(f"Created {len(all_chunks)} chunks total")
        return all_chunks

    def save_chunks_to_json(self, chunks: List[DocumentChunk], output_file: str):
        """Save chunks to JSON file"""
        chunks_data = []

        for chunk in chunks:
            chunk_data = {
                'content': chunk.content,
                'tokens': chunk.tokens,
                'metadata': asdict(chunk.metadata)
            }
            chunks_data.append(chunk_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(chunks)} chunks to {output_file}")

    def get_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        stats = {
            'total_chunks': len(chunks),
            'avg_token_count': sum(chunk.metadata.token_count for chunk in chunks) / len(chunks) if chunks else 0,
            'modules': {},
            'categories': {},
            'total_tokens': sum(chunk.metadata.token_count for chunk in chunks)
        }

        for chunk in chunks:
            # Module statistics
            if chunk.metadata.module not in stats['modules']:
                stats['modules'][chunk.metadata.module] = 0
            stats['modules'][chunk.metadata.module] += 1

            # Category statistics
            if chunk.metadata.category not in stats['categories']:
                stats['categories'][chunk.metadata.category] = 0
            stats['categories'][chunk.metadata.category] += 1

        return stats


def main():
    """Main function to run the chunking process"""
    import argparse

    parser = argparse.ArgumentParser(description='Chunk RIOT OS documentation for RAG')
    parser.add_argument('doc_dir', help='Directory containing RIOT documentation')
    parser.add_argument('--output', '-o', default='riot_chunks.json', help='Output JSON file')
    parser.add_argument('--chunk-size', type=int, default=800, help='Chunk size in tokens')
    parser.add_argument('--overlap-size', type=int, default=150, help='Overlap size in tokens')
    parser.add_argument('--scan-only', action='store_true', help='Only scan directory structure, don\'t process')
    parser.add_argument('--file-types', nargs='+', default=['html', 'md', 'rst', 'txt'],
                        help='File extensions to process (without dots)')

    args = parser.parse_args()

    # Verify directory exists
    doc_path = Path(args.doc_dir)
    if not doc_path.exists():
        print(f"Error: Directory {args.doc_dir} does not exist!")
        return

    if args.scan_only:
        print(f"Scanning directory structure: {doc_path}")
        scan_directory_structure(doc_path)
        return

    # Initialize chunker
    chunker = RIOTDocumentationChunker(
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size
    )

    # Update file patterns if specified
    if args.file_types != ['html', 'md', 'rst', 'txt']:
        print(f"Using custom file types: {args.file_types}")

    # Process documentation
    chunks = chunker.process_directory(args.doc_dir)

    if not chunks:
        print("No chunks were created! Please check your directory structure.")
        return

    # Save to JSON
    chunker.save_chunks_to_json(chunks, args.output)

    # Print statistics
    stats = chunker.get_statistics(chunks)
    print("\n=== Statistics ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average tokens per chunk: {stats['avg_token_count']:.1f}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Modules: {stats['modules']}")
    print(f"Categories: {stats['categories']}")


def scan_directory_structure(doc_path: Path, max_depth: int = 3):
    """Scan and display directory structure"""
    print(f"\nDirectory structure for: {doc_path}")
    print("=" * 50)

    def print_tree(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return

        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")

            if item.is_dir() and depth < max_depth:
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, depth + 1)

    print_tree(doc_path)

    # Count files by type
    file_counts = {}
    for file_path in doc_path.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            file_counts[ext] = file_counts.get(ext, 0) + 1

    print(f"\nFile type summary:")
    for ext, count in sorted(file_counts.items()):
        print(f"  {ext or '(no extension)'}: {count} files")


if __name__ == "__main__":
    main()