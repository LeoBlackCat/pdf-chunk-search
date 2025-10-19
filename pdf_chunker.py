#!/usr/bin/env python3
"""
PDF Chunker - Extract and chunk content from PDFs and plain text files.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from chunker import split_text, token_count
from pdf_extractor import extract_text_from_pdf


def extract_and_chunk(
    input_file: str,
    output_file: str,
    chunk_size: int = 256,
    chunk_overlap: Optional[int] = None,
    strategy: str = "smart",
) -> None:
    """
    Extract content from a file and split it into chunks.
    
    Args:
        input_file: Path to the input file (PDF, TXT, or Markdown)
        output_file: Path to the output file (one chunk per line)
        chunk_size: Maximum tokens per chunk (default: 256)
        chunk_overlap: Token overlap between chunks (default: 30 tokens)
        strategy: Chunking strategy ("smart", "sentence", "llama", or "langchain")
    """
    # Validate input file exists
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Extracting content from: {input_file}")
    text_content = _load_text(input_file)
    if not text_content:
        print("Warning: No text content extracted from file")
        return
    print(f"Extracted {len(text_content)} characters")
    print(f"Estimated tokens: {token_count(text_content)}")
    
    # Generate output filenames
    output_path = Path(output_file)
    extracted_file = output_path.parent / f"{output_path.stem}_extracted.txt"
    chunks_file = output_path.parent / f"{output_path.stem}_chunks.txt"
    
    # Write raw extracted text
    print(f"\nWriting extracted text to: {extracted_file}")
    with open(extracted_file, "w", encoding="utf-8") as f:
        f.write(text_content)
    print(f"✓ Successfully wrote extracted text to {extracted_file}")
    
    # Split into chunks
    if strategy == "sentence":
        effective_overlap = 0
    else:
        overlap_input = chunk_overlap if chunk_overlap is not None else 30
        effective_overlap = max(0, min(overlap_input, max(0, chunk_size - 1)))
    print(f"\nChunking with size={chunk_size}, overlap={effective_overlap}, strategy={strategy}")
    chunks = split_text(
        text_content,
        chunk_size=chunk_size,
        chunk_overlap=effective_overlap if strategy != "sentence" else 0,
        strategy=strategy,
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Write chunks to output file (one per line)
    print(f"\nWriting chunks to: {chunks_file}")
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    
    print(f"✓ Successfully wrote {len(chunks)} chunks to {chunks_file}")
    
    # Print statistics
    chunk_sizes = [token_count(c) for c in chunks]
    print(f"\nChunk statistics:")
    print(f"  Min tokens: {min(chunk_sizes)}")
    print(f"  Max tokens: {max(chunk_sizes)}")
    print(f"  Avg tokens: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
    
    print(f"\n📄 Output files:")
    print(f"  Raw text: {extracted_file}")
    print(f"  Chunks:   {chunks_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and chunk content from PDFs and plain text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf output.txt
  %(prog)s document.pdf output.txt --chunk-size 1000
  %(prog)s document.pdf output.txt --chunk-size 800 --chunk-overlap 100
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input file path (PDF, TXT, or Markdown)"
    )
    
    parser.add_argument(
        "output_file",
        help="Output file path (one chunk per line)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Maximum tokens per chunk (default: 256)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=30,
        help="Token overlap between chunks (default: 30)"
    )

    parser.add_argument(
        "--strategy",
        choices=["smart", "sentence", "llama", "langchain"],
        default="smart",
        help=(
            "Chunking strategy to apply (default: smart; "
            "use 'llama' for LlamaIndex SentenceSplitter or 'langchain' for RecursiveCharacterTextSplitter)"
        )
    )
    
    args = parser.parse_args()
    
    try:
        extract_and_chunk(
            args.input_file,
            args.output_file,
            args.chunk_size,
            args.chunk_overlap,
            args.strategy,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def _load_text(input_file: str) -> str:
    path = Path(input_file)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(str(path))

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    raise ValueError(
        f"Unsupported file extension '{suffix}'. Only PDF and plain text files are supported."
    )


if __name__ == "__main__":
    main()
