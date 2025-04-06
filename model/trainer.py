"""
This script fine-tunes the model using LoRA (Low-Rank Adaptation) 
to keep memory usage low, leveraging the instruct-answer dataset.
"""
# llm_pdf_finetuner/model/trainer.py
from transformers import TrainingArguments, Trainer as HFTrainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from typing import Tuple

class Trainer:
    """Fine-tunes an LLM using LoRA on an instruct-answer dataset."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, dataset_path: str, output_dir: str):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token
    
    def prepare_dataset(self):
        """Load and preprocess the instruct-answer dataset."""
        dataset = load_dataset("json", data_files=self.dataset_path)
        
        def preprocess_function(examples):
            inputs = [f"### Instruction: {instr}\n### Response: {resp}" 
                     for instr, resp in zip(examples["instruction"], examples["response"])]
            tokenized = self.tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        return dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "response"])
    
    def configure_lora(self):
        """Apply LoRA to the model."""
        lora_config = LoraConfig(
            r=16,  # Rank of adaptation
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Phi-1.5 specific layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def train(self):
        """Run the fine-tuning process."""
        dataset = self.prepare_dataset()
        train_dataset = dataset["train"]
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,  # Small batch size for 16GB RAM
            gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,  # Mixed precision for efficiency
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False
        )
        
        trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model fine-tuned and saved to {self.output_dir}")

if __name__ == "__main__":
    from model_loader import ModelLoader
    
    loader = ModelLoader()
    model, tokenizer = loader.load_model_and_tokenizer()
    trainer = Trainer(model, tokenizer, "output/instruct_dataset.json", "fine_tuned_model")
    trainer.configure_lora()
    trainer.train()