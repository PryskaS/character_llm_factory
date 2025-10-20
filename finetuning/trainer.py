import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

def main():
    """Main function to orchestrate the fine-tuning process."""
    
    # --- 1. Model and Tokenizer Setup ---
    print("Setting up model and tokenizer...")
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Data Loading and Preparation (Tokenization) ---
    # This is where we transform raw text into a structured format
    # that the model's neural network can process.
    print("Loading and tokenizing dataset...")
    
    # Load the dataset from our text file
    dataset = load_dataset("text", data_files={"train": "rick_corpus.txt"})

    # Tokenization function to apply to each example
    def tokenize_function(examples):
        # We process the text in batches for efficiency
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    # Apply the tokenization to the entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # The DataCollator is a helper that forms batches of data for training.
    # mlm=False means we are doing Causal Language Modeling (predicting the next word).
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 3. Training Configuration ---
    # TrainingArguments is the "control panel" for our fine-tuning job.
    # It defines all the hyperparameters and settings.
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir="./rick-llm-distilgpt2", # Directory to save the trained model
        overwrite_output_dir=True,
        num_train_epochs=3,                # Number of times to train on the full dataset
        per_device_train_batch_size=4,     # Number of examples per training step
        save_steps=10_000,                 # How often to save a checkpoint
        save_total_limit=2,                # Only keep the last 2 checkpoints
        prediction_loss_only=True,
        logging_steps=100,                 # How often to log training progress
    )

    # --- 4. Trainer Initialization ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    # --- 5. Start Fine-Tuning ---
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete!")

    # --- 6. Save the Final Model ---
    print("Saving the final model...")
    final_model_path = "./rick-llm-final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    main()