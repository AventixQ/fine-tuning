### CODE FOR LOCAL FINE-TUNING ON GPU

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datasets import Dataset
from input_tokenizer import tokenize_function as tokenize
from transformers import Trainer, TrainingArguments
import torch
from excel2json import excel_to_json

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
torch.cuda.empty_cache()

##### CHOOSE YOUR MODEL (to some of them you need acces from HF)

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2", torch_dtype=torch.float16)
#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", torch_dtype=torch.float16)
#tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")


tokenizer.pad_token = tokenizer.eos_token
#model.gradient_checkpointing_enable()


excel_file = 'examples-2000.xlsx'  ## Set an excel file name with data
json_file = 'input_data.json'  ## Set a json file name for your LLM fine-tuning data
excel_to_json(excel_file, json_file)
with open(json_file, 'r') as f:
    data = json.load(f)


dataset = Dataset.from_list(data)

tokenized_datasets = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

print(tokenized_datasets)

training_args = TrainingArguments(
    output_dir="./results",
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