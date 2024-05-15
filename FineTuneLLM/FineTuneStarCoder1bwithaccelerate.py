from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator

def main():
    model_name = "bigcode/starcoderbase-1b"
    dataset_name = "smangrul/hf-stack-v1"
    dataset_split = "train"

    # Initialize the accelerator
    accelerator = Accelerator()

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the pad_token to the eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Define the maximum length for the sequences
    max_length = 512

    # Load the dataset
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Print the dataset column names to inspect them
    print("Dataset columns:", dataset.column_names)

    # Tokenization function with defined max_length
    def tokenize_function(examples):
        return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8)

    # Set up training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=32,  # Increase batch size to use more memory
        per_device_eval_batch_size=32,   # Increase evaluation batch size
        num_train_epochs=3,
        learning_rate=5e-5,
        output_dir='./results',
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        dataloader_num_workers=8,  # Use more CPU cores for data loading
    )

    # Data collator to handle padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Prepare everything with `accelerate`
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        trainer.model,
        trainer.optimizer,
        trainer.get_train_dataloader(),
        trainer.lr_scheduler,
    )

    # Training loop
    trainer.train()

    # Save the model
    trainer.save_model()

if __name__ == "__main__":
    main()
