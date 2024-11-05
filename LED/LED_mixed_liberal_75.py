import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import logging
from datasets import Dataset, load_dataset
from itertools import chain
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    LEDForConditionalGeneration,
    LEDTokenizer,
    Seq2SeqTrainingArguments,
    Trainer
)
import os
from huggingface_hub import login, HfApi

# Configure logging
logging.basicConfig(filename='LED_mixed_liberal_75_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Check if CUDA is available and list available devices
# if torch.cuda.is_available():
#     num_gpus = torch.cuda.device_count()
#     logging.info("CUDA is available")
#     logging.info(f"Number of GPUs: {num_gpus}")
#     for i in range(num_gpus):
#         logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
#     # Set CUDA_VISIBLE_DEVICES to use GPU 0 if it exists
#     if num_gpus > 1:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#         device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#         logging.info(f"Using GPU Device: {device}")
# else:
#     logging.warning("CUDA is not available")

# Replace 'your_huggingface_token' with your actual token
login(token='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')

dataset = load_dataset("ahmed275/mixed_dataset_temporal_liberal_75")

# Load the tokenizer
tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')

def preprocess_function(examples):
    logging.info(f"Preprocessing {len(examples)} examples")
    inputs = examples['opinionOfTheCourt']
    targets = examples['syllabus']
    model_inputs = tokenizer(inputs, max_length=16384, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=1024, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']

    batch = {}
    batch["input_ids"] = model_inputs.input_ids
    batch["attention_mask"] = model_inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = labels.input_ids

    # We have to make sure that the PAD token is ignored
    # -100 for loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

# Apply the preprocessing function to the datasets
logging.info("Tokenizing datasets")
tokenized_train_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
tokenized_val_dataset = dataset['validation'].map(preprocess_function, batched=True, remove_columns=dataset['validation'].column_names)
# tokenized_test_dataset = dataset['test'].map(preprocess_function, batched=True, remove_columns=dataset['test'].column_names)

# Load the model
model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')

# Set generate hyperparameters
model.config.num_beams = 2
model.config.max_length = 1024
model.config.min_length = 256
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./srv/results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Train the model
logging.info("Starting training")
trainer.train()

# Evaluate the model
logging.info("Evaluating the model")
results = trainer.evaluate()
logging.info(f"Evaluation results: {results}")

# # Generate summaries
# def generate_summary(opinion):
#     inputs = tokenizer(opinion, return_tensors='pt', max_length=16384, truncation=True, padding='max_length')
#     summary_ids = model.generate(inputs['input_ids'].to("cuda"), max_length=1024, num_beams=4, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Assuming `model` is your trained model
model.save_pretrained("led-base-16384/model")
tokenizer.save_pretrained("led-base-16384/model")

api = HfApi()

# Replace 'your-username/your-model-name' with your desired repository name
repo_id = "ahmed275/SS-LED_mixed_liberal_75"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
# Upload the model
api.upload_folder(
    folder_path="led-base-16384/model",
    repo_id=repo_id,
    repo_type="model"
)

# # Test the model on a sample opinion
# sample_opinion = df['opinionOfTheCourt'].iloc[0]
# logging.info(f"Testing model on sample opinion: {sample_opinion}")
# logging.info(f"Generated Summary: {generate_summary(sample_opinion)}")