import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from huggingface_hub import login

login(token="hf_xXXXXXXXX")

model = AutoModelForSequenceClassification.from_pretrained("./finetuned-v3")
tokenizer = AutoTokenizer.from_pretrained("./finetuned-v3")

model.push_to_hub("dkkh8788/finetuned_classification_model_v3")
tokenizer.push_to_hub("dkkh8788/finetuned_classification_model_v3")

"""
# Testing the model
classifier = pipeline("text-classification", model='dkkh8788/finetuned_classification_model_v3',
                      tokenizer='dkkh8788/finetuned_classification_model_v3')

new_texts = [
    "can you tell me status of Order id: 123123111",
    "what is the status of my Order ID: 123456987",
    "I need help setting up my iphone 16?"
]

for text in new_texts:
    res = classifier(text)
    print(res)
    print(f"Text: {text}\nPredicted Label: {res[0]['label']}\n")
"""
