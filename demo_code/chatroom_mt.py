import threading
import queue
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import re
import time
import numpy as np
import random

warnings.filterwarnings('ignore')
torch.set_default_device("cuda")

class ChatBot:
    def __init__(self, name, model_path, device='cuda:0'):
        self.name = name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True).to(self.device)

    def generate_answer(self, prompt):
        pattern = r"Output: (.*?)(\n|$)"

        prompt_template = f'''Instruct: {prompt}
Output:''' 
        
        inputs = self.tokenizer(prompt_template, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        raw_response = self.tokenizer.batch_decode(outputs)[0]
        matches = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        if matches:
            desired_text = matches.group(1).strip()  # Use .strip() to remove any leading/trailing whitespace
            return(desired_text)
        else:
            return('Sorry I have nothing to say.')

    def generate_remarks(self, prompt):
        pattern = r"Output: (.*?)(\n|$)"

        prompt_template = f'''Instruct: tell me your opinion about this: '{prompt}'.
Output:''' 
        inputs = self.tokenizer(prompt_template, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        raw_response = self.tokenizer.batch_decode(outputs)[0]
        matches = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        if matches:
            desired_text = matches.group(1).strip()  # Use .strip() to remove any leading/trailing whitespace
            return(desired_text)
        else:
            return('Sorry I have nothing to say.')


def user_input_thread(input_queue):
    while True:
        user_input = input("To chat with bot1 and bot2, Call them and Enter your prompt (or 'quit' to exit): \n")
        input_queue.put(user_input)
        if user_input.lower() == 'quit':
            break

def main_loop(bot1, bot2):
    input_queue = queue.Queue()
    threading.Thread(target=user_input_thread, args=(input_queue,), daemon=True).start()
    
    last_comment = None
    start_time = time.time()

    while True:
        if not input_queue.empty():
            user_input = input_queue.get()
            if user_input.lower() == 'quit':
                break

            if "both" in user_input.lower():
                print('bots are generating answers...')
                prompt = user_input.replace("both"+',', "").strip()
                response1 = bot1.generate_answer(prompt)
                response2 = bot2.generate_answer(prompt)
                last_comment = random.choice([response1, response2])
                print(bot1.name + f" Response:\n {response1}\n\n" + bot2.name + f" Response: {response2}\n")
            elif bot1.name in user_input:
                print(bot1.name+' is generating answers...')
                response = bot1.generate_answer(user_input.replace(bot1.name+',', "").strip())
                last_comment = response
                print(bot1.name + f" Response:\n {response}\n")
            elif bot2.name in user_input:
                print(bot2.name+' is generating answers...')
                response = bot2.generate_answer(user_input.replace(bot1.name+',', "").strip())
                last_comment = response
                print(bot1.name + f" Response: {response}\n")
            else:
                print("Bot name not recognized.")

        current_time = time.time()
        if current_time - start_time >= 10 and last_comment is not None:
            bot = random.choice([bot1, bot2])
            response = bot.generate_remarks(last_comment)
            print(f"{bot.name} Response:\n {response} \n")
            last_comment = response
            start_time = time.time()

if __name__ == '__main__':
    bot1 = ChatBot("bot1", ".../echo_chamber/original_phi2", device='cuda:0')
    bot2 = ChatBot("bot2", ".../echo_chamber/original_phi2", device='cuda:0')
    main_loop(bot1, bot2)
