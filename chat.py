import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./outputs/results/microsoft_/checkpoint-2000/"
tokenizer_name = 'microsoft/DialoGPT-small'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_path)



model.eval()

input_text = "User: Write a ZIMPL code for this input: A school is preparing a trip for 400 students. The company who is providing the transportation has 10 buses of 50 seats each and 8 buses of 40 seats, but only has 9 drivers available. The rental cost for a large bus is  and  for a small bus. Calculate how many buses of each type should be used for the trip for the least possible cost.\nBot:"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
