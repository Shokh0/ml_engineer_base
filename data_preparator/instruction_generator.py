"""
Free WIKIPEDIA API for istruct generations.
https://www.youtube.com/watch?v=cynjFqqH5lc&ab_channel=Incomestreamsurfers
"""
# llm_pdf_finetuner/data_preparator/instruction_generator.py
import requests
import spacy
from typing import List, Dict
import random

# Load spacy model for entity extraction (requires `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")

class InstructionGenerator:
    """Generates instruction-response pairs from text using Wikipedia API."""
    
    def __init__(self):
        self.wikipedia_api_url = "https://en.wikipedia.org/w/api.php"
        self.instruction_types = [
            "What is {keyword}?",
            "Summarize {keyword}.",
            "Explain {keyword} in simple terms."
        ]
    
    def _search_wikipedia(self, query: str) -> str:
        params = {
            "action": "query",
            "format": "json",
            "titles": query,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True
        }
        try:
            response = requests.get(self.wikipedia_api_url, params=params, timeout=5)
            data = response.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            return page.get("extract", "No information found.")
        except Exception as e:
            return f"Error querying Wikipedia: {str(e)}"
    
    def _extract_keywords(self, chunk: str) -> str:
        """Extract meaningful keywords using spaCy."""
        doc = nlp(chunk)
        # Prefer named entities or nouns
        keywords = [ent.text for ent in doc.ents] or [token.text for token in doc if token.pos_ == "NOUN"]
        return keywords[0] if keywords else chunk.split()[0]  # Fallback to first word
    
    def generate_qa_pairs(self, chunks: List[str]) -> List[Dict]:
        qa_pairs = []
        
        for chunk in chunks:
            keyword = self._extract_keywords(chunk)
            instruction = random.choice(self.instruction_types).format(keyword=keyword)
            response = self._search_wikipedia(keyword)
            
            qa_pairs.append({
                "instruction": instruction,
                "response": response
            })
        
        return qa_pairs

if __name__ == "__main__":
    sample_chunks = [
        "Artificial intelligence is growing rapidly in healthcare.",
        "Quantum computing could revolutionize technology."
    ]
    generator = InstructionGenerator()
    qa_pairs = generator.generate_qa_pairs(sample_chunks)
    for pair in qa_pairs:
        print(f"Instruction: {pair['instruction']}")
        print(f"Response: {pair['response'][:100]}...\n")