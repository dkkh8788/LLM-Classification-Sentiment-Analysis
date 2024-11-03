import torch
import pandas as pd
from datasets import Dataset
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
 
# Replace with your API Key
client = InferenceClient(api_key="hf_XXXXXX")

classifier = pipeline("text-classification", model='dkkh8788/finetuned_classification_model_v3',
                      tokenizer='dkkh8788/finetuned_classification_model_v3', device='mps')

def generate_response(feedback, sentiment):

    if sentiment == "POSITIVE":
        prompt = f"Generate an appropriate positive response in maximum 50 words to the customer thanking and expressing gratitude."
    else:
        prompt = f"Generate an appropriate empathetic response in maximum 50 words to the customer apologizing for the inconvenience and offering assistance."

    messages = [
        { "role": "assistant", "content": f"{feedback}\n\nQuestion: {prompt}" },
    ]
 
    output = client.chat.completions.create(
        model="microsoft/Phi-3.5-mini-instruct",
        messages=messages,
        stream=True,
        temperature=0.5,
        max_tokens=1024,
        top_p=0.7
    )
 
    # Collect all chunks in a list and join them after the loop
    full_response = []
 
    for chunk in output:
        full_response.append(chunk.choices[0].delta.content)
 
    # Join and return the entire response as a single string
    return ("".join(full_response))

def perform_sentiment_analysis(text):  
  sentiment_analyzer = pipeline(
                        'sentiment-analysis', model=
                            'distilbert/distilbert-base-uncased-finetuned-sst-2-english', device="mps")

  result = sentiment_analyzer(text)
  return result[0]['label']

def filter(text, category):
  if category == "feedback":
    sentiment = perform_sentiment_analysis(text)

    # Further process the query based on sentiment
    response = generate_response(text, sentiment)
    print(response)


# Classify new text samples
new_texts = [
    "can you tell me status of Order id: 123123111",
    "what is the status of my Order ID: 123456987",
    "does iphone 10 Pro Max have camera zoom fearues ?",
    "My order with Id: 127865331 was delayed. can i know where it is now ?",
    "latest iphone is just amazing!",
    "i need help with iphone 10 keypad set up",
    "how to set up my iphone 11",
    "is my order id 123987222 status in portal showing transit or delivered ?",
    "the experience with Iphones are best. they have less bugs and works smooth",
    "How can i set up the message forwarding in iphone 12",
    "we loved the purchase. the experience was top notch.",
    "we have not liked the service. it could have been better",
    "the best thing about iphones are the tag they have. i liked it very much.",
    "how to set up the battery in iphone ?",
    "i order iphone 12 on tuesday. my order id is 123999888. could you please update me on the order status ?",
    "can you tell me my order 1234 position ?",
    "Can you tell me my order number 1234 State State",
    "Can you tell me my order number 1 2 3 4 State",
    "My order is 1 2 3 4",
    "1234 status",
    "bad experience with buying MAC M1"
]

for text in new_texts:
    print("\n=======INPUT=======\n")
    result = classifier(text)
    output_text = result[0]['label']
    input_text = text
    # print(f"Classification Result: {result}")
    print(f"Text: {text}. [ Predicted Class: {result[0]['label']} ]\n")
    filter(input_text, output_text)
