# RAG for RIOT OS

A Retrieval-Augmented Generation (RAG) toolchain for the [RIOT operating system](https://github.com/RIOT-OS/RIOT). This repository helps you build local vector search indices over RIOT's documentation and example code to power LLM-assisted development, code generation, and explanations.


---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Documentation RAG](#documentation-rag)

   * [1. Clone RIOT](#1-clone-riot)
   * [2. Generate Documentation](#2-generate-documentation)
   * [3. Chunk & Embed Documentation](#3-chunk--embed-documentation)
   * [4. Query the RAG](#4-query-the-rag)
3. [Autoencoder (Optional)](#autoencoder-optional)
4. [Code RAG (Examples Directory)](#code-rag-examples-directory)
5. [Combining the RAGS](#Best-of-two-worlds)
6. [Ease of use](#GUI)

---

## Prerequisites

* Python 3.8+
* [Doxygen](https://www.doxygen.nl/) to generate API documentation
* Required pip packages:

  ```bash
  pip install chromadb sentence-transformers torch numpy tqdm scikit-learn flask flask-cors beautifulsoup4 tiktoken langchain
  ```

---

## Documentation RAG

Leverage RIOT's API docs for retrieval-augmented prompts.

### 1. Clone RIOT

```bash
git clone https://github.com/RIOT-OS/RIOT.git
cd RIOT
```

### 2. Generate Documentation

Use Doxygen to build HTML or XML docs locally:

```bash
doxygen Doxyfile
# Output will be in ./doc or ./html by default
```

### 3. Chunk & Embed Documentation

1. **Chunk** the generated docs:

   ```bash
   python3 RIOTDocuChunker2.py path/to/RIOT/doc/html
   ```

   * Produces `riot_chunks.json` containing overlapping text chunks and metadata.

2. **Embed** chunks into a vector database:

   ```bash
   python3 RIOTRRAGDocuDB3.py riot_chunks.json
   ```

   * Creates a ChromaDB at `./riot_vector_db` (default path, configurable).

### 4. Query the RAG

Retrieve relevant documentation snippets for any query:

```bash
python3 RIORDocuRAGRequest2.py "<your query>"
```

The script returns:

* Your original user query
* Top matching documentation chunks
* A ready-to-use prompt template for your LLM

---

## Autoencoder (Optional)

Compress embeddings to speed up search and potentially improve relevance.

1. **Standard Autoencoder**:

   ```bash
   python3 AutoencoderRIOT2.py
   ```
2. **Triplet Autoencoder** (margin-based grouping):

   ```bash
   python3 AutoencoderRIOTTriplet.py --epochs 100 --lambda-triplet 5.0 --margin 1.5
   ```

**Note:** Compare performance with the uncompressed RAG to evaluate impact.

To query with compressed vectors:

```bash
python3 RIORDocuRAGRequestCompressed.py "<your query>"
python3 RIORDocuRAGRequestCompressedTriplet.py "<your query>"
```

---

## Code RAG (Examples Directory)

Perform RAG over RIOT's `examples/` codebase.

1. **Set examples directory** in `chunker.py` (line 11):

   ```python
   EXAMPLES_DIR = "/path/to/RIOT/examples"
   ```
2. **Chunk** the examples:

   ```bash
   python3 chunker.py
   ```
3. **Embed** the chunks:

   ```bash
   python3 embedder.py
   ```
4. **Query** the example RAG:

   ```bash
   python3 request.py "<your query>"
   ```

> Warning: If no example matches your query, results may be irrelevant. Use alongside Documentation RAG.

---

## Best of two worlds 
If you saved all python files in one common directory (like in this repository), you can try to run a request which searches both rags and returns the combined results:

 ```bash
   python3 RIOTRequestCombined.py "<your query>"
   ```
## GUI
If you saved all python files in one common directory (like in this repository), you can try to run the python GUI, here you can easily choose which RAG you want to use specify your query and also specify parameters
 ```bash
   python3 RAGSystemGUI.py
   ```
Alternatively you can also use the more good looking GUI, 
First run
 ```bash
   python3 RAGGUIServer.py
   ```
Then go to http://127.0.0.1:5000/ here you should now see the website version of the GUI

