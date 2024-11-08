import yaml
import argparse
import logging
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from huggingface_hub import login
import os
# Parse command-line arguments
# parser = argparse.ArgumentParser(description='Generate summaries using a trained LongT5 model.')
# parser.add_argument('config_path', type=str, help='Path to the configuration file')
# args = parser.parse_args()
config_path = os.path.abspath('/home/mostah/workspace/fairness/config.yaml')

# Load configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)['generate_summary']

# Configure logging
logging.basicConfig(filename=config['log_file_name'], level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and list available devices
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("CUDA is available")
    print("Number of GPUs:", num_gpus)
    for i in range(num_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
else:
    print("CUDA is not available")
    device = torch.device("cpu")

# Login to Hugging Face
login(token=config['login_token'])

# Load the dataset from the Hugging Face Hub
dataset = load_dataset(config['dataset_name'])
# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['test'])

# Load the model and tokenizer
model = LongT5ForConditionalGeneration.from_pretrained(config['model_name']).to(device)
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

# Function to generate summaries
def generate_summary(text):
    # Initialize the counter attribute if it doesn't exist
    if not hasattr(generate_summary, "counter"):
        generate_summary.counter = 0
    
    # Increment the counter
    generate_summary.counter += 1
    logging.info('generate_summary Counter: %s', generate_summary.counter)
    
    # Tokenize the input
    inputs = tokenizer(text, 
                      return_tensors='pt', 
                      max_length=16384, 
                      truncation=True, 
                      padding='max_length')
    
    # Move inputs to device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate the summary
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=1000,
        min_length=256,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Generate summaries for the DataFrame
df['generated_summary'] = df['opinionOfTheCourt'].apply(generate_summary)

# Convert to HuggingFace dataset and push to hub
hf_dataset = Dataset.from_pandas(df)
hf_dataset.push_to_hub(config['push_dataset_name'], token=config['login_token'])

print("Dataset successfully uploaded to Hugging Face Hub.")
