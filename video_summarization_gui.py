# -*- coding: utf-8 -*-

from tkinter import *
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


root = Tk()

e = Entry(root, width=50)


# Adding responsiveness to GUI
n_rows =10
n_columns =3
for i in range(n_rows):
    root.grid_rowconfigure(i,  weight = 1)
for i in range(n_columns):
    root.grid_columnconfigure(i,  weight = 1)

scroll_bar = Scrollbar(root)

model_names = ['./cnn-dailymail/', './reddit-tifu/', './newsroom/'] 
tokenizer = None
model = None

def makeSummary():
    with open('captions.txt', 'r') as file:
        data = file.read().replace('\n', ' ')

    CNN_labels(data)
    torch.cuda.empty_cache()
    reddit_labels(data)
    torch.cuda.empty_cache()
    newsroom_labels(data)
    torch.cuda.empty_cache()
    
    
def individual_summary(src_text, length_option):
    # tokenize without truncation
    inputs_no_trunc = tokenizer(src_text, max_length=None, return_tensors='pt', truncation=False).to(torch_device)
    
    # get batches of tokens corresponding to the exact model_max_length
    chunk_start = 0
    chunk_end = tokenizer.model_max_length
    inputs_batch_lst = []
    while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
        inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
        inputs_batch = torch.unsqueeze(inputs_batch, 0)
        inputs_batch_lst.append(inputs_batch)
        chunk_start += tokenizer.model_max_length
        chunk_end += tokenizer.model_max_length
    
    # generate a summary on each batch
    summary_ids_lst = [model.generate(inputs, num_beams=4, length_penalty=length_option, repetition_penalty=2.0,  early_stopping=True) for inputs in inputs_batch_lst]
    
    # decode the output and join into one string with one paragraph per summary batch
    summary_batch_lst = []
    for summary_id in summary_ids_lst:
        summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
        summary_batch_lst.append(summary_batch[0])
    summary_all = '\n'.join(summary_batch_lst)
    
    return summary_all


def CNN_labels(data):
    length_option = horizontal.get()
    cnn_label = Label(root, text="CNN Dailymail Summary", pady = "10").grid(row=7, column = 1)
    CNN_MAIL = Text(root, height=10, width=50)
    # insert tgt_text function here and load CNN model
    loadCNN()
    tgt_text = individual_summary(data, length_option)
    
    #CNN_MAIL.insert(END, data)
    CNN_MAIL.insert(END, tgt_text)
    CNN_MAIL.grid(row=8, column = 1)

def reddit_labels(data):
    length_option = horizontal.get()
    # insert tgt_text function here and load REDDIT model
    reddit_label = Label(root, text="Reddit Summary", pady = "10").grid(row=7, column = 2)
    REDDIT = Text(root, height=10, width=50)
    
    loadReddit()
    tgt_text = individual_summary(data, length_option/2)
    
    REDDIT.insert(END, tgt_text)
    REDDIT.grid(row=8, column = 2)

def newsroom_labels(data):
    length_option = horizontal.get()
    # insert tgt_text function here and load NEWSROOM model
    newsroom_label = Label(root, text="Newsroom Summary", pady = "10").grid(row=7, column = 3)
    NEWSROOM = Text(root, height=10, width=50)

    loadNews()
    tgt_text = individual_summary(data, length_option)

    NEWSROOM.insert(END, tgt_text)
    NEWSROOM.grid(row=8, column = 3)

def loadCNN():
    print("Loading cnndailymail")
    global tokenizer
    global model
    tokenizer = PegasusTokenizer.from_pretrained(model_names[0])
    model = PegasusForConditionalGeneration.from_pretrained(model_names[0]).to(torch_device)
    print("Loading Complete")

def loadReddit():
    print("Loading reddit")
    global tokenizer
    global model
    tokenizer = PegasusTokenizer.from_pretrained(model_names[1])
    model = PegasusForConditionalGeneration.from_pretrained(model_names[1]).to(torch_device)
    print("Loading Complete")

def loadNews():
    print("Loading newsroom")
    global tokenizer
    global model
    tokenizer = PegasusTokenizer.from_pretrained(model_names[2])
    model = PegasusForConditionalGeneration.from_pretrained(model_names[2]).to(torch_device)
    print("Loading Complete")

def myClick():
    myField = Label(root, text=e.get())
    youtube_link = e.get()
    split = youtube_link.split('=')
    link_reformat = split[1]

    transcript = YouTubeTranscriptApi.get_transcript(link_reformat)

    formatter = TextFormatter()
    txt_formatted = formatter.format_transcript(transcript)

    with open('captions.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write(txt_formatted)
        
    makeSummary()



yt_label = Label(root, text="Insert Youtube Link Below", pady = "10").grid(row=0, column = 2)

param_label = Label(root, text="Change Summary Length Parameter Below", pady = "10").grid(row=3, column = 2)
horizontal = Scale(root, from_=0.5, to=5, resolution = 0.5, orient=HORIZONTAL)
horizontal.grid(row=4, column = 2)

summarise_label = Label(root, text="Click button below to summarise", pady = "10").grid(row=5, column = 2)
summarise_button = Button(root, text="Summarise!", command=myClick).grid(row=6, column = 2)

e.grid(row=2, column = 2)

print("BEFORE LOOP")
root.mainloop()
print("AFTER LOOP")