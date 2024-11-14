### CODE FOR LOCAL FINE-TUNING ON GPU

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datasets import Dataset
from input_tokenizer import tokenize_function as tokenize
from transformers import Trainer, TrainingArguments
import torch
from excel2json import excel_to_json
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
    model_name = "meta-llama/CodeLlama-7b-Python-hf"
else:
    model_name = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Model {model_name} was loaded correctly.")

file_names = create_output_dir_name(model_name)


##### CHOOSE YOUR MODEL (to some of them you need acces from HF)

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
#tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
#model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")
#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
#tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Python-hf")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-Python-hf")

tokenizer.pad_token = tokenizer.eos_token
#model.gradient_checkpointing_enable()


excel_file = 'examples-2000.xlsx'  ## Set an excel file name with data
json_file = f'input_data_{file_names}.json'  ## Set a json file name for your LLM fine-tuning data
excel_to_json(excel_file, json_file)
with open(json_file, 'r') as f:
    data = json.load(f)


dataset = Dataset.from_list(data)

tokenized_datasets = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

print(tokenized_datasets)

output_dir = f"./results/{file_names}"
print(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    gradient_accumulation_steps=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

torch.cuda.empty_cache()
trainer.train()