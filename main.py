# llm_pdf_finetuner/main.py (example)
# llm_pdf_finetuner/main.py
from data_preparator.pdf_loader import PDFLoader
from data_preparator.cleaner import TextCleaner
from data_preparator.dataset_formatter import DatasetFormatter
from model.model_loader import ModelLoader
from model.trainer import Trainer

def main():
    # Data preparation
    loader = PDFLoader("sample_pdfs")
    pdf_contents = loader.load_all_pdfs()
    cleaner = TextCleaner(max_chunk_length=512)
    all_chunks = [chunk for _, text in pdf_contents for chunk in cleaner.chunk_text(text)]
    formatter = DatasetFormatter("output", use_api=True)
    formatter.format_and_save(all_chunks)

    # Model training
    loader = ModelLoader()
    model, tokenizer = loader.load_model_and_tokenizer()
    trainer = Trainer(model, tokenizer, "output/instruct_dataset.json", "fine_tuned_model")
    trainer.configure_lora()
    trainer.train()

if __name__ == "__main__":
    main()