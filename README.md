# PDF Chunk Search

A Python tool for extracting content from PDFs and other documents, then splitting the text into manageable chunks for processing with LLMs or other text analysis tools.

## Features

- Extract text from PDFs with PyMuPDF (plus direct reading for plain text files)
- Smart text chunking with configurable size and overlap
- Optional LlamaIndex-powered sentence splitter and LangChain recursive splitter modes
- Automatic MLX embeddings generated for every chunk
- FAISS-powered vector search across the generated chunks
- Batch multiple documents into a single chunk/embedding set
- Optional OpenAI-powered cleanup (gpt-4o-mini) when `--ai-clean` is used and `OPENAI_API_KEY` is provided
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
python pdf_chunker.py output/law law1.pdf
```

You can list multiple PDFs (or text files) to consolidate them into a single chunk set:

```bash
python pdf_chunker.py output/law law1.pdf law2.pdf law3.pdf
```

To include OpenAI cleanup (requires `OPENAI_API_KEY`):

```bash
python pdf_chunker.py output/law law1.pdf law2.pdf --ai-clean
```

Each run:
- Extracts text from every input file
- Writes the cleaned text for each document to `output/law_####_<name>_extracted.txt`
- Splits the cleaned text into ~256-token chunks (default `sentence` strategy)
- Stores chunk metadata in `output/law_chunks.jsonl`
- Stores document-level metadata in `output/law_docs.jsonl`
- Saves embeddings for all chunks in `output/law_embeddings.npy`

### Multiple Inputs

You can pass any mix of PDFs, Markdown, or plain-text files. They will be processed sequentially and merged into one metadata/embedding set. Chunk IDs remain unique and align with both the JSONL metadata and the saved `.npy` embedding matrix, so you can rebuild a FAISS index or re-run searches at any time.

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
  --embeddings output_embeddings.npy \
  --query "Monteverdi opera reforms" \
  --top-k 3 \
  --with-context
```

This prints the best-matching chunk(s) with chunk IDs, doc IDs, per-document chunk indexes, page numbers, spans, and optional neighbors for quick inspection.

### Command-Line Options

- `output_prefix` - Base name for output files (creates `*_extracted.txt`, `*_docs.jsonl`, `*_chunks.jsonl`, `*_embeddings.npy`)
- `input_files` - One or more input files (PDF, TXT, or Markdown)
- `--chunk-size` - Maximum tokens per chunk (default: 256)
- `--chunk-overlap` - Token overlap between consecutive chunks (default: 30 tokens)
- `--strategy` - Chunking approach to use (`smart`, `sentence`, `llama`, or `langchain`; default: `sentence`)
- `--ai-clean` - Enable OpenAI cleanup (requires `OPENAI_API_KEY`)

Add an `.env` file with `OPENAI_API_KEY=...` (or export the variable in your shell) **and** pass `--ai-clean` to enable OpenAI-powered cleanup during extraction.

### Output Format

Each run emits three artifacts:

1. **`*_extracted.txt`** – Cleaned extracted text (exactly what was chunked and embedded).
2. **`*_docs.jsonl`** – One JSON object per source with provenance (`doc_id`, extractor info, pipeline version, etc.).
3. **`*_chunks.jsonl`** – One JSON object per chunk containing the embedded text, span metadata, model details, and overlap statistics (embedding vectors are stored separately in `*_embeddings.npy`).

Chunk records use `span_scope="doc_final"` so `char_start/char_end` are offsets within the cleaned (and optionally AI-polished) document text. If `clean_span` is present it references the pre-AI cleaned text (`clean_span_scope="doc_clean"`). Both document and chunk records include `schema_version`, `pipeline_version`, and `metric` fields for future migrations.

Each chunk line retains only metadata (no inline vectors). Embeddings are stored in `*_embeddings.npy`, in the same order as the JSONL records, which makes it easy to rebuild FAISS indices or migrate to a different vector database.
Chunk IDs are monotonically increasing integers (`id`) and double as FAISS vector IDs. Chunk metadata now contains precise page ranges (`page`, `page_end`, and `pages`) computed directly from the per-page text, so no heuristic page estimates.
Document records include matching provenance (e.g., `pipeline_version="extract=pymupdf@1.26.5;clean=python-clean-v1+gpt-4o-mini;chunk=sentence-v1(256/0);embed=mlx-community/all-MiniLM-L6-v2-4bit@384"`).

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

If you pass `--ai-clean` and an `OPENAI_API_KEY` is available (for example via a local `.env` file), the raw extracted text is additionally passed through OpenAI's `gpt-4o-mini` for light cleanup before chunking.

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
python pdf_chunker.py chunks/research research_paper.pdf
```

Output:
```
Extracting content from: research_paper.pdf
Extracted 45230 characters
Estimated tokens: 11307

✓ Wrote cleaned text to chunks/research_0000_research_paper_extracted.txt

Chunking with size=500, overlap=75, strategy=smart
Created 24 chunks

Generating embeddings for 24 chunks
Embeddings shape: (24, 384)

📄 Output files:
  Document metadata: chunks/research_docs.jsonl
  Chunks metadata:  chunks/research_chunks.jsonl
  Embeddings:       chunks/research_embeddings.npy

Chunk statistics:
  Min tokens: 387
  Max tokens: 500
  Avg tokens: 476.2
```

### Example 2: Large documents with bigger chunks

```bash
python pdf_chunker.py chunks/book large_book.pdf --chunk-size 2000 --chunk-overlap 200
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
