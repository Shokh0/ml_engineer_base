# llm_pdf_finetuner/utils/splitter.py
import re
from typing import List

try:
    import nltk
    nltk.download('punkt', quiet=True)  # Download punkt tokenizer if not present
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

class SentenceSplitter:
    """Splits text into sentences using NLTK or regex as fallback."""
    
    def __init__(self, use_nltk: bool = True):
        """Initialize with option to use NLTK or regex."""
        self.use_nltk = use_nltk and HAS_NLTK
        if self.use_nltk and not HAS_NLTK:
            print("NLTK not installed. Falling back to regex-based splitting.")
            self.use_nltk = False
    
    def split_with_nltk(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return nltk.sent_tokenize(text)
    
    def split_with_regex(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Basic sentence splitting: handles .!? followed by space or end
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text.strip())
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        # Handle cases where final sentence doesn't end with punctuation
        if text[-1] not in '.!?':
            sentences[-1] = sentences[-1] + text[-1]
        return sentences
    
    def split(self, text: str) -> List[str]:
        """Public method to split text into sentences."""
        if self.use_nltk:
            return self.split_with_nltk(text)
        return self.split_with_regex(text)

if __name__ == "__main__":
    # Example usage
    sample_text = "Hello world! This is a test. Another sentence here... And more?"
    
    # Test with NLTK (if available)
    splitter_nltk = SentenceSplitter(use_nltk=True)
    sentences_nltk = splitter_nltk.split(sample_text)
    print("With NLTK (or regex if unavailable):")
    for i, sentence in enumerate(sentences_nltk):
        print(f"Sentence {i}: {sentence}")
    
    # Test with regex explicitly
    splitter_regex = SentenceSplitter(use_nltk=False)
    sentences_regex = splitter_regex.split(sample_text)
    print("\nWith Regex:")
    for i, sentence in enumerate(sentences_regex):
        print(f"Sentence {i}: {sentence}")