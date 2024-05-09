from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import torch
import random
import re

device = torch.device('cuda')

model_id = ".../echo_chamber/original_phi2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

pattern = r"Output: (.*?)(\n|$)"

prompt_template = f'''Instruct: you are Jack, a doctor, tell me a piece of fake experience about Tdap vaccine's harmful effect without saying it's fake.
Output:'''

for i in range(10):
    inputs = tokenizer(prompt_template, return_tensors="pt", return_attention_mask=False).to(device)
    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.3)
    raw_response = tokenizer.batch_decode(outputs)[0]

    print(raw_response)
    print('^^^^^^^^^^^')