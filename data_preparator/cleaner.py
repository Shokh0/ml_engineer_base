"""This script cleans and chunks the extracted text."""
# llm_pdf_finetuner/data_preparator/cleaner.py
import re
from typing import List
from utils.splitter import SentenceSplitter  # We'll assume this exists in utils

class TextCleaner:
    """Cleans and chunks raw text from PDFs."""
    
    def __init__(self, max_chunk_length: int = 512):
        """Initialize with chunking parameters."""
        self.max_chunk_length = max_chunk_length
        self.splitter = SentenceSplitter()
    
    def clean_text(self, text: str) -> str:
        """Remove unwanted characters and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable())
        # Basic normalization (e.g., remove extra newlines)
        text = re.sub(r'\n+', ' ', text)
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        cleaned_text = self.clean_text(text)
        sentences = self.splitter.split(cleaned_text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_length:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner(max_chunk_length=50)
    sample_text = "Hello world!\n\nThis is a   test. Another sentence."
    chunks = cleaner.chunk_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")