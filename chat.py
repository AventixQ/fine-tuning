import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model.eval()

input_text = "User: There are 5 people in the room. How many people is in the room?\nBot:"
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
