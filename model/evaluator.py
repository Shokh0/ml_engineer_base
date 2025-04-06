"""This script evaluates the fine-tuned model’s performance."""
# llm_pdf_finetuner/model/evaluator.py
from transformers import pipeline
import torch

class Evaluator:
    """Evaluates a fine-tuned model on sample inputs."""
    
    def __init__(self, model, tokenizer, device: str = "mps" if torch.backends.mps.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=self.device)
    
    def evaluate(self, instruction: str, max_length: int = 100) -> str:
        """Generate a response for a given instruction."""
        prompt = f"### Instruction: {instruction}\n### Response: "
        response = self.generator(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]
        return response.split("### Response: ")[-1].strip()

if __name__ == "__main__":
    from model_loader import ModelLoader
    
    loader = ModelLoader()
    model, tokenizer = loader.load_peft_model("fine_tuned_model")
    evaluator = Evaluator(model, tokenizer)
    print(evaluator.evaluate("What is artificial intelligence?"))