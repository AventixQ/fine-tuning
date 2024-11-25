from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
import sys
import os
        
def tokenize(input_data, tokenizer):
    inputs = input_data["input"]
    outputs = input_data["output"]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length").input_ids
    
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    
    model_inputs["labels"] = labels
    return model_inputs

def create_output_dir_name(model_name: str) -> str:
    parts = model_name.split('/')
    last_part = parts[-1]
    prefix = parts[0].split('-')[0] if '-' in parts[0] else parts[0]
    folder_name = f"{prefix}_{last_part}"[:10]
    
    return folder_name

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
torch.cuda.empty_cache()

if len(sys.argv) != 2:
    model_name = "microsoft/DialoGPT-small"

else:
    model_name = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


tokenizer.pad_token = tokenizer.eos_token


print(f"Model {model_name} was loaded correctly.")
file_names = create_output_dir_name(model_name)

dataset = load_dataset("Tamiza/zimpl_data", split="train")

tokenized_datasets = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

print(tokenized_datasets)

output_dir = f"./results2/{file_names}"
print(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

torch.cuda.empty_cache()
trainer.train()