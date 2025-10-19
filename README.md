# PDF Chunker

A Python tool for extracting content from PDFs and other documents, then splitting the text into manageable chunks for processing with LLMs or other text analysis tools.

## Features

- Extract text from PDFs with PyMuPDF (plus direct reading for plain text files)
- Smart text chunking with configurable size and overlap
- Optional LlamaIndex-powered sentence splitter and LangChain recursive splitter modes
- Token-aware splitting (uses mlx-embeddings for efficient token counts)
- Hierarchical splitting strategy (paragraphs → lines → sentences → words)
- Output format: one chunk per line for easy processing

## Installation

1. Install required dependencies:

```bash
pip install PyMuPDF mlx-embeddings llama-index langchain langchain-text-splitters
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
- Write raw extracted text to `output_extracted.txt`
- Split it into chunks of ~256 tokens each
- Write chunks to `output_chunks.txt` (one per line)

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

### Command-Line Options

- `input_file` - Path to input file (PDF, TXT, or Markdown)
- `output_file` - Base name for output files (will create `*_extracted.txt` and `*_chunks.txt`)
- `--chunk-size` - Maximum tokens per chunk (default: 256)
- `--chunk-overlap` - Token overlap between consecutive chunks (default: 30 tokens)
- `--strategy` - Chunking approach to use (`smart`, `sentence`, `llama`, or `langchain`)

### Output Format

The tool creates two output files:

1. **`*_extracted.txt`** - Raw extracted text from the document (preserves original formatting)
2. **`*_chunks.txt`** - Chunked text with one chunk per line

The chunks file has newlines within chunks escaped as `\n` so each chunk occupies exactly one line. This makes it easy to process with standard Unix tools:

```bash
# Count chunks
wc -l output_chunks.txt

# View first chunk
head -n 1 output_chunks.txt

# View raw extracted text
less output_extracted.txt

# Process each chunk
while IFS= read -r chunk; do
    echo "Processing: ${chunk:0:50}..."
done < output_chunks.txt
```

## How It Works

### 1. Content Extraction

The tool uses a lightweight PyMuPDF helper (`pdf_extractor.py`) to read PDFs and applies cleaning routines inspired by `clean_pdf.py`:

```python
from pdf_extractor import extract_text_from_pdf

text = extract_text_from_pdf("filename.pdf")
```

Plain `.txt` and `.md` files are read directly from disk without additional processing.

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

Chunks are written to the output file with:
- One chunk per line
- Newlines escaped as `\n`
- Statistics printed to console

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

Writing extracted text to: chunks_extracted.txt
✓ Successfully wrote extracted text to chunks_extracted.txt

Chunking with size=500, overlap=75
Created 24 chunks

Writing chunks to: chunks_chunks.txt
✓ Successfully wrote 24 chunks to chunks_chunks.txt

Chunk statistics:
  Min tokens: 387
  Max tokens: 500
  Avg tokens: 476.2

📄 Output files:
  Raw text: chunks_extracted.txt
  Chunks:   chunks_chunks.txt
```

### Example 2: Large documents with bigger chunks

```bash
python pdf_chunker.py large_book.pdf book_chunks.txt --chunk-size 2000 --chunk-overlap 200
```

### Example 3: Process the output

```python
# Read raw extracted text
with open("output_extracted.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Read chunks back in Python
with open("output_chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip().replace("\\n", "\n") for line in f]

# Process each chunk
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
- **mlx-embeddings** - Token counting
- **llama-index** - Optional sentence-splitter strategy (`--strategy llama`)
- **langchain** / **langchain-text-splitters** - Optional RecursiveCharacterTextSplitter (`--strategy langchain`)

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
