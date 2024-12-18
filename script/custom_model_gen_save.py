#!pip install transformers datasets evaluate pandas huggingface_hub

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from huggingface_hub import InferenceClient

# Load the training dataset
df = pd.read_csv("shuffled_sample_dataset.csv")
dataset = Dataset.from_pandas(df)

# Load the test dataset
test_df = pd.read_csv("shuffled_sample_test_dataset.csv")
test_dataset = Dataset.from_pandas(test_df)

# Define custom labels
labels = ["order_id_status", "feedback", "product_setup"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}


# Map dataset labels to IDs
def encode_labels(example):
    example['label'] = label2id[example['label']]
    return example

dataset = dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)

# Tokenize the dataset using tokenizer pre-trained on a model called "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenizer convert text data into numerical representations.
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Load pre-trained BERT model from the HF
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(labels), # number of output classes
    id2label=id2label, # maps label id to label name
    label2id=label2id  # maps label name to label id
)
"""
# Define training arguments
#training_args = TrainingArguments(
    output_dir='./finetuned-v2',
    evaluation_strategy="epoch", # evaluation after each epoch
    learning_rate=2e-5, # learning rate
    per_device_train_batch_size=16, # training size at once
    per_device_eval_batch_size=12, # test size at once
    num_train_epochs=3, # number of epocs
    weight_decay=0.01,
)
"""

# Define training arguments
training_args = TrainingArguments(
    output_dir='./finetuned-v3',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=14,
    per_device_eval_batch_size=10,
    num_train_epochs=4,
    weight_decay=0.01,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./finetuned-v3")
tokenizer.save_pretrained("./finetuned-v3")

# Evaluate the model
results = trainer.evaluate()
print(results)

