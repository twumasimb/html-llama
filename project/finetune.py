import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


from datasets import load_from_disk

def load_model_and_tokenizer():
    model_name = "meta-llama/CodeLlama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization for 8-bit instead of 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = f"""
### Input:
{data_point["prompt"]}

### Response:
```
{data_point["code"]}
```
"""
    return tokenizer(
        full_prompt,
        padding=True,
        truncation=True,
        max_length=2048  # removed return_tensors="pt"
    )

def prepare_dataset(tokenizer):
    dataset = load_from_disk("./final_dataset/final_dataset_3600")
    
    # Add dataset validation
    if not all(col in dataset["train"].column_names for col in ["prompt", "code"]):
        raise ValueError("Dataset must contain 'prompt' and 'code' columns")
    
    tokenized_dataset = dataset.map(
        generate_and_tokenize_prompt,
        fn_kwargs={"tokenizer": tokenizer},
        batched=False,  # Process one example at a time
        num_proc=4,  # Use multiple processes for speed
        remove_columns=dataset["train"].column_names
    )
    
    print(f"Original dataset sizes - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    print(f"Tokenized dataset sizes - Train: {len(tokenized_dataset['train'])}, Test: {len(tokenized_dataset['test'])}")
    
    return tokenized_dataset

def main():
    # Initialize model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    save_path = "models"
    os.makedirs(save_path, exist_ok=True)
    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer)
    print(f"Size of tokenized dataset: {len(tokenized_dataset['train'])}")
    # Update training arguments
    training_args = TrainingArguments(
        output_dir=f"{save_path}/results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-4,
        fp16=True,
        gradient_checkpointing=True,
        report_to="tensorboard"
    )
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model(f"{save_path}/codellama-a11y-{len(tokenized_dataset['train'])}")
    
    # Merge LoRA weights with the base model
    model = model.merge_and_unload()
    model.save_pretrained("./codellama-a11y-merged-3600")

if __name__ == "__main__":
    main()