import os
import fitz  # PyMuPDF
import pymupdf4llm as pdf
from pathlib import Path
from llama_index.readers.file import PDFReader

def extract_text(fpath: str, ftype: str) -> str:
    """Extract text from a file.

    Args:
        fpath: Path to the file.
        ftype: File extension/type (e.g. 'pdf', 'txt').

    Returns:
        Extracted text as a string.
    """
    if ftype == "pdf":
        return _pdf_to_text(fpath)
    # naive text load; code kept as-is for chunker
    return Path(fpath).read_text(errors="ignore")


def _pdf_to_text(fpath: str) -> str:
    doc = pdf.to_markdown(fpath)
    return doc

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the extract_text function on a file.")
    parser.add_argument("--fpath", type=str, default="./test_data/table.pdf", help="Path to input file (PDF or text).")

    args = parser.parse_args()

    fpath = args.fpath
    ftype = os.path.splitext(fpath)[1].lower().lstrip(".")

    text = extract_text(fpath, ftype)
    docs = PDFReader().load_data(fpath)[0]
    text_llama = docs.get_text()
    print("=" * 80)
    print(f"Extracted text from {fpath} ({ftype}):")
    print("=" * 80)
    print(text)  # print first 2000 chars for inspection
    print("----------------------------------------------")
    print(text_llama)
    