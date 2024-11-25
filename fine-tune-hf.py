from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import torch
import sys
import os

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
torch.cuda.empty_cache()

def create_output_dir_name(model_name: str) -> str:
    parts = model_name.split('/')
    last_part = parts[-1]
    prefix = parts[0].split('-')[0] if '-' in parts[0] else parts[0]
    folder_name = f"{prefix}_{last_part}"[:10]
    
    return folder_name

if len(sys.argv) != 2:
    model_name = "microsoft/DialoGPT-small"
else:
    model_name = sys.argv[1]

dataset = load_dataset("Tamiza/zimpl_data", split="train")

dataset = dataset.rename_columns({"input": "instruction", "output": "response"})

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model {model_name} was loaded correctly.")
file_names = create_output_dir_name(model_name)

def formatting_prompts_func(example):
    return [f"### Please, provide ZIMPL code for this task:\n{example['instruction']}\n### Answer:\n{example['response']}"]

from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

output_dir = f"./results/{file_names}"
print(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = SFTConfig(
    output_dir=output_dir,
    eval_strategy="no",
    save_steps=500,
    #logging_dir=f"./results/{file_names}/logs",
    report_to="none",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    load_best_model_at_end=False
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

torch.cuda.empty_cache()

#import logging
#logging.basicConfig(level=logging.DEBUG)

trainer.train()

    
print("Saving model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Saved.")

