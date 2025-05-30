
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
from sklearn.model_selection import train_test_split
import numpy as np


# Load dataset
dataset = load_dataset("json", data_files="health_insurance_dataset_new_n.jsonl")["train"]


# Format prompt
def format_prompt(example):
    example["text"] = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return example

dataset = dataset.map(format_prompt)


# Split into train and validation (90/10 split)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]


# Load tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# Tokenize datasets
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_train = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)


# Load base model (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)


# Apply LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)


# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-lora-health-insurance",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)


# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator
)

# Train
trainer.train()


# Save final model
model.save_pretrained("./mistral-lora-attempt-new-2")
tokenizer.save_pretrained("./mistral-lora-attempt-new-2")
print("Training complete. Final LoRA adapter and tokenizer saved.")
