"""This script loads the Phi-1.5 model with quantization for efficiency."""
# llm_pdf_finetuner/model/model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # For LoRA if applied
import torch
from typing import Tuple

class ModelLoader:
    """Loads a pretrained LLM and tokenizer with quantization."""
    
    def __init__(self, model_name: str = "microsoft/phi-1_5", use_quantization: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # M1/M2 GPU or CPU
    
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer with 4-bit quantization if specified."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.use_quantization:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,  # 4-bit quantization
                device_map="auto",  # Automatically map to available device
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            ).to(self.device)
        
        return model, tokenizer
    
    def load_peft_model(self, peft_checkpoint: str) -> PeftModel:
        """Load a fine-tuned model with PEFT (e.g., LoRA) weights."""
        base_model, tokenizer = self.load_model_and_tokenizer()
        model = PeftModel.from_pretrained(base_model, peft_checkpoint)
        return model, tokenizer

if __name__ == "__main__":
    loader = ModelLoader()
    model, tokenizer = loader.load_model_and_tokenizer()
    print(f"Model loaded on {loader.device}: {model.config.model_type}")