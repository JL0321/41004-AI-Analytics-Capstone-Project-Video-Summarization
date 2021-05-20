# 41004-AI-Analytics-Capstone-Project-Video-Summarization

This is a gui for youtube video summarization using google's pegasus summarization model.

to use, download the pretrained models from this link:https://drive.google.com/drive/folders/169723nPUCy1LP-jbGp_fVTS599g5Pa6P?usp=sharing and place them in the same directory as the video_summarization_gui.py file then run the file.

the packages required are:
  - tkinter
  - youtube_transcript_api
  - pytorch
  - transformers from huggingface

It will run on the gpu if cuda is available, otherwise on cpu

The video link must contain english captions enabled for the youtube api to fetch the transcript
