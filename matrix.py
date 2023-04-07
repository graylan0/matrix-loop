import json
import torch
import time
from tkinter import Tk, Label, Entry, Button, Text, Scrollbar, Y, RIGHT, END
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import threading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125m')

# add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# set padding token id to the id of the padding token
model.config.pad_token_id = tokenizer.pad_token_id

stop_loop = False

# Load settings and bot token from the JSON file
with open('settings.json', 'r') as settings_file:
    settings = json.load(settings_file)

# Get the loop count value
loop_count = settings['loop_count']

# Load trideque matrix from the JSON file
with open('trideque.json', 'r') as trideque_file:
    trideque = json.load(trideque_file)

def generate_chunks(prompt, chunk_size=1500):
    words = prompt.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def gpt3_generate(model, tokenizer, chunk, max_length=2000, time_limit=50.0):
    start_time = time.time()

    inputs = tokenizer.encode(chunk, return_tensors='pt', truncation=True, max_length=512).to(device)
    attention_mask = inputs.ne(tokenizer.pad_token_id).float().to(device)
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, max_time=time_limit, attention_mask=attention_mask)

    response = tokenizer.decode(outputs[0])
    end_time = time.time()

    return response, end_time - start_time

def send_chunks(trideque_point, loop_count=-1):
    global stop_loop
    total_time = 0.0
    repetition = 0

    if 0 <= trideque_point < len(trideque):
        while (loop_count == -1 or repetition < loop_count) and not stop_loop:
            for topic in trideque[trideque_point]:
                prompt_chunks = generate_chunks(topic)
                for chunk in prompt_chunks:
                    gpt3_response, response_time = gpt3_generate(model, tokenizer, chunk)
                    total_time += response_time
                    output_text.insert(END, f"{topic}: {gpt3_response}\n")

            repetition += 1

        output_text.insert(END, f"Total response time: {total_time:.2f} seconds.\n")
    else:
        output_text.insert(END, "Invalid trideque point. Please enter a valid index.\n")


def on_generate_click():
    trideque_point = int(trideque_point_input.get())
    threading.Thread(target=send_chunks, args=(trideque_point, loop_count)).start()


def on_stop_loop_click():
    global stop_loop
    stop_loop = True

# GUI setup
root = Tk()
root.title("TheMatrix")
root.geometry("954x800")

# set the background color for the GUI window
root.config(background='black')
Label(root, text="Point:", fg="green", bg="black", font=("Courier", 14)).grid(row=2, column=0, sticky="W")

trideque_point_input = Entry(root, width=10)
trideque_point_input.grid(row=3, column=0)

Label(root, text="Enter input:", fg="green", bg="black", font=("Courier", 14)).grid(row=0, column=0, sticky="W")

input_text = Entry(root, width=100)
input_text.grid(row=1, column=0)

Button(root, text="Generate", command=on_generate_click, bg="green", fg="black", font=("Courier", 14)).grid(row=1, column=1)
Button(root, text="Stop Loop", command=on_stop_loop_click, bg="green", fg="black", font=("Courier", 14)).grid(row=1, column=2)

output_text = Text(root, wrap="word", width=80, height=20, bg="#0a0a0a", fg="#00ff00", font=("Courier", 14))
output_text.grid(row=2, column=0, columnspan=6, padx=10, pady=10)

scrollbar = Scrollbar(root, command=output_text.yview)
scrollbar.grid(row=2, column=6, sticky="ns")
output_text.config(yscrollcommand=scrollbar.set)

root.mainloop()