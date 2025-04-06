# llm_pdf_finetuner/data_preparator/dataset_formatter.py
from typing import List, Dict
import json
from pathlib import Path
from .instruction_generator import InstructionGenerator

class DatasetFormatter:
    """Formats cleaned text into an instruct-answer dataset for fine-tuning."""
    
    def __init__(self, output_dir: str, use_api: bool = True):
        """Initialize with output directory and API usage flag."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = InstructionGenerator(use_api=use_api)
    
    def format_for_instruction_tuning(self, chunks: List[str]) -> List[Dict]:
        """Format chunks into instruction-response pairs."""
        return self.generator.generate_qa_pairs(chunks)
    
    def save_dataset(self, dataset: List[Dict], filename: str = "instruct_dataset.json"):
        """Save the formatted dataset to a file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {output_path}")
    
    def format_and_save(self, chunks: List[str], filename: str = "instruct_dataset.json"):
        """Format chunks and save to file."""
        dataset = self.format_for_instruction_tuning(chunks)
        self.save_dataset(dataset, filename)

if __name__ == "__main__":
    # Example usage
    formatter = DatasetFormatter("output", use_api=True)
    sample_chunks = ["Machine learning is advancing.", "Blockchain technology is secure."]
    formatter.format_and_save(sample_chunks)