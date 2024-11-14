import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages = [
    { 'role': 'user', 'content': "Write a ZIMPL code for this task: Niki holds two part-time jobs, Job I and Job II. She never wants to work more than a total of 12 hours a week. She has determined that for every hour she works at Job I, she needs 2 hours of preparation time, and for every hour she works at Job II, she needs one hour of preparation time, and she cannot spend more than 16 hours for preparation. If Niki makes $40 an hour at Job I, and $30 an hour at Job II, how many hours should she work per week at each job to maximize her income?"}
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.config.eos_token_id


if isinstance(inputs, torch.Tensor):
    inputs = {'input_ids': inputs}

inputs = {key: value.to(model.device) for key, value in inputs.items()}

outputs = model.generate(inputs["input_ids"], max_new_tokens=512, do_sample=True, temperature = 0)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
