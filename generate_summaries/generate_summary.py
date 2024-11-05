


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

# Configure logging
logging.basicConfig(filename='generated_summary.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and list available devices
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

# Load the dataset from the Hugging Face Hub
dataset = load_dataset("ahmed275/opinions_dataset_temporal")
# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['test'])

# Load the model and tokenizer from Hugging Face
model = AutoModelForSeq2SeqLM.from_pretrained("ahmed275/SS-LED_mixed_liberal_75").to(device)
tokenizer = AutoTokenizer.from_pretrained("ahmed275/SS-LED_mixed_liberal_75")

# Function to generate summaries
def generate_summary(opinion):
    # Initialize the counter attribute if it doesn't exist
    if not hasattr(generate_summary, "counter"):
        generate_summary.counter = 0
    
    # Increment the counter
    generate_summary.counter += 1
    
    # Log the counter value
    logging.info('generate_summary Counter: %s', generate_summary.counter)
    # Tokenize the input and move it to the specified device
    inputs = tokenizer(opinion, return_tensors='pt', max_length=16383, truncation=True, padding='max_length')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], max_length=1000, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Generate summaries for the DataFrame
df['generated_summary'] = df['opinionOfTheCourt'].apply(generate_summary)
hf_dataset = Dataset.from_pandas(df)
hf_dataset.push_to_hub("ahmed275/opinions_dataset_temporal_test_generated_summaries_mixed_liberal_75", token='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')

print("Dataset successfully uploaded to Hugging Face Hub.")