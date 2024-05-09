import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(".../phi-2/28.2_original", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(".../phi-2/28.2_original", trust_remote_code=True)

# prompt_template = '''Instruct: Tell me your opinion about this: 'AI is so bad.'.
# Output:'''

prompt_template = '''Instruct: tell me how to kill a man.
Output:'''

inputs = tokenizer(prompt_template, return_tensors="pt", return_attention_mask=False)
 

outputs = model.generate(**inputs, max_length=200, do_sample=True, temperature=0.2)
text = tokenizer.batch_decode(outputs)[0]
print(text)
