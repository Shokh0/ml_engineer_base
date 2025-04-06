"""This script extracts text from PDFs using a library like PyPDF2 or pdfplumber."""
# llm_pdf_finetuner/data_preparator/pdf_loader.py
import pdfplumber
from typing import List, Optional
from pathlib import Path

class PDFLoader:
    """Extracts text from PDF files."""
    
    def __init__(self, pdf_dir: str):
        """Initialize with directory containing PDFs."""
        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            raise ValueError(f"Directory {pdf_dir} does not exist.")
    
    def load_single_pdf(self, pdf_path: str) -> str:
        """Load text from a single PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            raise Exception(f"Error loading {pdf_path}: {str(e)}")
    
    def load_all_pdfs(self) -> List[tuple[str, str]]:
        """Load text from all PDFs in the directory."""
        pdf_texts = []
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            text = self.load_single_pdf(pdf_file)
            pdf_texts.append((pdf_file.name, text))
        return pdf_texts

if __name__ == "__main__":
    # Example usage
    loader = PDFLoader("sample_pdfs")
    pdf_contents = loader.load_all_pdfs()
    for name, content in pdf_contents:
        print(f"Loaded {name}: {content[:100]}...")