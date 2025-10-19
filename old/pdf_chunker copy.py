#!/usr/bin/env python3
"""
PDF Chunker - Extract and chunk content from PDFs and other documents
"""
import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import content_core as cc
except ImportError:
    print("Error: content_core module not found. Please install it first.")
    print("Try: pip install content-core")
    sys.exit(1)

from chunker import split_text, token_count


async def extract_and_chunk(
    input_file: str,
    output_file: str,
    chunk_size: int = 500,
    chunk_overlap: Optional[int] = None,
) -> None:
    """
    Extract content from a file and split it into chunks.
    
    Args:
        input_file: Path to the input file (PDF, DOCX, TXT, etc.)
        output_file: Path to the output file (one chunk per line)
        chunk_size: Maximum tokens per chunk (default: 500)
        chunk_overlap: Token overlap between chunks (default: 15% of chunk_size)
    """
    # Validate input file exists
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Extracting content from: {input_file}")
    
    # Extract content using content_core
    try:
        # content_core expects a dict with file_path key
        result = await cc.extract({"file_path": input_file})
        
        # Result is a ProcessSourceOutput object with a 'content' attribute
        if hasattr(result, 'content'):
            text_content = result.content
        elif isinstance(result, dict):
            text_content = result.get("content", "") or result.get("text", "")
        else:
            # Fallback - shouldn't happen
            text_content = str(result)
        
        if not text_content:
            print("Warning: No text content extracted from file")
            return
            
        print(f"Extracted {len(text_content)} characters")
        print(f"Estimated tokens: {token_count(text_content)}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract content: {e}")
    
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
    print(f"\nChunking with size={chunk_size}, overlap={chunk_overlap or int(chunk_size * 0.15)}")
    chunks = split_text(
        text_content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Write chunks to output file (one per line)
    print(f"\nWriting chunks to: {chunks_file}")
    with open(chunks_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            # Escape newlines within chunks so each chunk is on one line
            escaped_chunk = chunk.replace("\n", "\\n").replace("\r", "\\r")
            f.write(escaped_chunk + "\n")
    
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
        description="Extract and chunk content from PDFs and other documents",
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
        help="Input file path (PDF, DOCX, TXT, etc.)"
    )
    
    parser.add_argument(
        "output_file",
        help="Output file path (one chunk per line)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum tokens per chunk (default: 500)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=30,
        help="Token overlap between chunks (default: 15%% of chunk-size)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(extract_and_chunk(
            args.input_file,
            args.output_file,
            args.chunk_size,
            args.chunk_overlap,
        ))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
