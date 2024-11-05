from itertools import product
import logging
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



from datasets import Dataset
from huggingface_hub import HfApi, HfFolder

import logging
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import StratifiedShuffleSplit
from huggingface_hub import login, HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from Levenshtein import distance as levenshtein_distance
import os


logging.basicConfig(filename='generated_summary_decoding.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load the dataset from the Hugging Face Hub
dataset = load_dataset("ahmed275/opinions_dataset_temporal")
# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['test'])

# Load the model and tokenizer from Hugging Face
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("CUDA is available")
    print("Number of GPUs:", num_gpus)
    for i in range(num_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Set CUDA_VISIBLE_DEVICES to use GPU 0 if it exists
    if num_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")
else:
    print("CUDA is not available")

# Replace 'your_huggingface_token' with your actual token
login(token='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')

model = AutoModelForSeq2SeqLM.from_pretrained("ahmed275/SS-LED_temporal").to(device)
tokenizer = AutoTokenizer.from_pretrained("ahmed275/SS-LED_temporal")

# Define parameter ranges
top_p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
temperature_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
top_k_values = [10, 25, 50, 75, 100, 250, 500, 1000, 1500, 2000]

# Function to generate summaries
def generate_summary(opinion, top_p=None, temperature=None, top_k=None):
    # Initialize the counter attribute if it doesn't exist
    if not hasattr(generate_summary, "counter"):
        generate_summary.counter = 0
    
    # Increment the counter
    generate_summary.counter += 1
    
    # Log the counter value
    logging.info('generate_summary Counter: %s', generate_summary.counter)
    
    # Tokenize the input and move it to the specified device
    inputs = tokenizer(opinion, return_tensors='pt', max_length=16384, truncation=True, padding='max_length')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Prepare the generation arguments
    generation_args = {
        'inputs': inputs['input_ids'],
        'max_length': 1024,
        'num_beams': 4,
        'early_stopping': True,
        'do_sample': True 
    }
    
    # Add optional parameters if they are provided
    if top_p is not None:
        generation_args['top_p'] = top_p
    if temperature is not None:
        generation_args['temperature'] = temperature
    if top_k is not None:
        generation_args['top_k'] = top_k
    
    # Generate the summary with specified parameters
    summary_ids = model.generate(**generation_args)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Iterate over top_p values
for top_p in top_p_values:
    logging.info(f'Testing with top_p={top_p}')
    df['generated_summary'] = df['opinionOfTheCourt'].apply(
        lambda opinion: generate_summary(opinion, top_p=top_p)
    )
    # Convert to Hugging Face Dataset and upload
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(f"ahmed275/decoding_summaries_top_p_{top_p}", token='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')
    logging.info(f"Dataset with top_p={top_p} successfully uploaded to Hugging Face Hub.")

# Iterate over temperature values
for temperature in temperature_values:
    logging.info(f'Testing with temperature={temperature}')
    df['generated_summary'] = df['opinionOfTheCourt'].apply(
        lambda opinion: generate_summary(opinion, temperature=temperature)
    )
    # Convert to Hugging Face Dataset and upload
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(f"ahmed275/decoding_summaries_temperature_{temperature}", token='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')
    logging.info(f"Dataset with temperature={temperature} successfully uploaded to Hugging Face Hub.")

# Iterate over top_k values
for top_k in top_k_values:
    logging.info(f'Testing with top_k={top_k}')
    df['generated_summary'] = df['opinionOfTheCourt'].apply(
        lambda opinion: generate_summary(opinion, top_k=top_k)
    )
    # Convert to Hugging Face Dataset and upload
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(f"ahmed275/decoding_summaries_top_k_{top_k}", token='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')
    logging.info(f"Dataset with top_k={top_k} successfully uploaded to Hugging Face Hub.")

logging.info("Testing and uploading completed.")