# PDF Chunker

A Python tool for extracting content from PDFs and other documents, then splitting the text into manageable chunks for processing with LLMs or other text analysis tools.

## Features

- Extract text from PDFs with PyMuPDF (plus direct reading for plain text files)
- Smart text chunking with configurable size and overlap
- Optional LlamaIndex-powered sentence splitter and LangChain recursive splitter modes
- Automatic MLX embeddings generated for every chunk
- FAISS-powered vector search across the generated chunks
- Optional OpenAI-powered cleanup when `OPENAI_API_KEY` is provided
- Token-aware splitting (uses mlx-embeddings for efficient token counts)
- Hierarchical splitting strategy (paragraphs → lines → sentences → words)
- Structured JSONL output (`*_docs.jsonl` + `*_chunks.jsonl`) for durable storage and re-embedding

## Installation

1. Install required dependencies:

```bash
pip install PyMuPDF mlx-embeddings numpy faiss-cpu llama-index langchain langchain-text-splitters openai python-dotenv
```

2. Make the script executable (optional):

```bash
chmod +x pdf_chunker.py
```

## Usage

### Basic Usage

```bash
python pdf_chunker.py input.pdf output.txt
```

This will:
- Extract text from `input.pdf`
- Write the cleaned text to `output_extracted.txt`
- Split it into chunks of ~256 tokens each
- Persist chunk metadata + embeddings to `output_chunks.jsonl`
- Write document-level provenance to `output_docs.jsonl`

### Advanced Usage

**Custom chunk size:**
```bash
python pdf_chunker.py document.pdf chunks.txt --chunk-size 1000
```

**Custom chunk size and overlap:**
```bash
python pdf_chunker.py document.pdf chunks.txt --chunk-size 800 --chunk-overlap 100
```

**Use the LlamaIndex splitter:**
```bash
python pdf_chunker.py document.pdf chunks.txt --strategy llama --chunk-size 512 --chunk-overlap 40
```

**Use the LangChain recursive splitter:**
```bash
python pdf_chunker.py document.pdf chunks.txt --strategy langchain --chunk-size 512 --chunk-overlap 40
```

### Search the Chunks

After chunking, you can query the embeddings with FAISS:

```bash
python chunk_search.py \
  --chunks output_chunks.jsonl \
  --query "Monteverdi opera reforms" \
  --top-k 3 \
  --with-context
```

This prints the best-matching chunk(s) with doc IDs, page numbers, spans, and optional neighbors for quick inspection.

### Command-Line Options

- `input_file` - Path to input file (PDF, TXT, or Markdown)
- `output_file` - Base name for output files (creates `*_extracted.txt`, `*_docs.jsonl`, and `*_chunks.jsonl`)
- `--chunk-size` - Maximum tokens per chunk (default: 256)
- `--chunk-overlap` - Token overlap between consecutive chunks (default: 30 tokens)
- `--strategy` - Chunking approach to use (`smart`, `sentence`, `llama`, or `langchain`; default: `sentence`)

Add an `.env` file with `OPENAI_API_KEY=...` (or export the variable in your shell) to enable OpenAI-powered cleanup during extraction.

### Output Format

Each run emits three artifacts:

1. **`*_extracted.txt`** – Cleaned extracted text (exactly what was chunked and embedded).
2. **`*_docs.jsonl`** – One JSON object per source with provenance (`doc_id`, extractor info, pipeline version, etc.).
3. **`*_chunks.jsonl`** – One JSON object per chunk containing the embedded text, spans, embeddings, model metadata, and overlap details.

Because the chunks are JSON, you can inspect or filter them with standard tools:

```bash
# Count chunks
jq -c '.' output_chunks.jsonl | wc -l

# View the first chunk record
head -n 1 output_chunks.jsonl | jq

# Read the cleaned extracted text
less output_extracted.txt
```

## How It Works

### 1. Content Extraction

The tool uses a lightweight PyMuPDF helper (`pdf_extractor.py`) to read PDFs and applies cleaning routines inspired by `clean_pdf.py`:

```python
from pdf_extractor import extract_text_from_pdf

result = extract_text_from_pdf("filename.pdf")
text = result.final_text  # cleaned (optionally AI-polished) text
```

Plain `.txt` and `.md` files are read directly from disk without additional processing.

If an `OPENAI_API_KEY` environment variable is present (for example via a local `.env` file), the raw extracted text is additionally passed through OpenAI's `gpt-5` for light cleanup before chunking.

### 2. Smart Chunking

The `chunker.py` module implements a recursive text splitting algorithm:

1. **Hierarchical splitting** - Tries to split on larger boundaries first:
   - Paragraphs (`\n\n`)
   - Lines (`\n`)
   - Sentences (`.`)
   - Clauses (`,`)
   - Words (` `)
   - Characters (last resort)

2. **Token-aware** - Uses mlx-embeddings for tokenizer-backed token counts (falls back to a character heuristic if unavailable)

3. **Overlap support** - Maintains context between chunks by including overlap from the previous chunk

4. **Greedy packing** - Efficiently packs text into chunks without exceeding the token limit

All strategies share a common separator hierarchy (double newline → newline → period → comma → space → Unicode specials → characters) to keep boundaries consistent.

If you select the `llama` strategy, the tool delegates chunking to `llama_index.core.node_parser.SentenceSplitter`, using the same chunk size and overlap budgets you provide. The `langchain` strategy leverages `langchain.text_splitter.RecursiveCharacterTextSplitter` with the same separator list for comparison.

### 3. Output

Each chunk is stored as a JSON object containing the embedded text, spans, embedding vector, and model metadata. This makes it straightforward to rebuild FAISS indices, audit changes, or migrate to new embedding models without re-running extraction.

## Examples

### Example 1: Process a PDF with default settings

```bash
python pdf_chunker.py research_paper.pdf chunks.txt
```

Output:
```
Extracting content from: research_paper.pdf
Extracted 45230 characters
Estimated tokens: 11307

✓ Wrote cleaned text to chunks_extracted.txt

Chunking with size=500, overlap=75, strategy=smart
Created 24 chunks

Generating embeddings for 24 chunks
Embeddings shape: (24, 384)

📄 Output files:
  Extracted:  chunks_extracted.txt
  Document metadata: chunks_docs.jsonl
  Chunks & embeddings: chunks_chunks.jsonl

Chunk statistics:
  Min tokens: 387
  Max tokens: 500
  Avg tokens: 476.2
```

### Example 2: Large documents with bigger chunks

```bash
python pdf_chunker.py large_book.pdf book_chunks.txt --chunk-size 2000 --chunk-overlap 200
```

### Example 3: Process the output

```python
import json
import numpy as np

chunks = []
embeddings = []
with open("output_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        chunks.append(record["text"])
        embeddings.append(record["embedding"])

embedding_matrix = np.array(embeddings, dtype=np.float32)

with open("output_extracted.txt", "r", encoding="utf-8") as f:
    extracted = f.read()

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {len(chunk)} chars")
    # Send to LLM, analyze, etc.
```

## Project Structure

```
.
├── pdf_chunker.py    # Main application script
├── chunker.py        # Text chunking logic
├── README.md         # This file
└── ccore.txt         # Sample input file
```

## Dependencies

- **PyMuPDF** - PDF extraction
- **mlx-embeddings** - Token counting and embeddings
- **faiss-cpu** - Vector search backend
- **numpy** - Embedding storage
- **llama-index** - Optional sentence-splitter strategy (`--strategy llama`)
- **langchain** / **langchain-text-splitters** - Optional RecursiveCharacterTextSplitter (`--strategy langchain`)
- **openai** / **python-dotenv** - Optional AI cleanup when `OPENAI_API_KEY` is supplied

## Troubleshooting

**"Cannot import fitz"**
```bash
pip install PyMuPDF
```

**"Unsupported file extension"**
Only `.pdf`, `.txt`, and `.md` files are supported out of the box. Convert other document types before processing.

**"No text content extracted from file"**
- Check if the file is readable
- Verify the file extension is one of `.pdf`, `.txt`, or `.md`
- Try opening the file manually to ensure it contains text

**Chunks are too large/small**
- Adjust `--chunk-size` parameter
- Note: Token count is approximate and may vary slightly

## License

This project is provided as-is for text processing and analysis tasks.
