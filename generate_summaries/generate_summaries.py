import yaml
import argparse
import logging
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate summaries using a trained model.')
parser.add_argument('config_path', type=str, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)['generate_summary']

# Configure logging
logging.basicConfig(filename=config['log_file_name'], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and list available devices
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("CUDA is available")
    print("Number of GPUs:", num_gpus)
    for i in range(num_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Set CUDA_VISIBLE_DEVICES to use GPU 0 if it exists
    if num_gpus > 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")
else:
    print("CUDA is not available")

# Login to Hugging Face
login(token=config['login_token'])

# Load the dataset from the Hugging Face Hub
dataset = load_dataset(config['dataset_name'])
# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['test'])

# Load the model and tokenizer using eval
model = eval(config['model']).to(device)
tokenizer = eval(config['tokenizer'])

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
hf_dataset.push_to_hub(config['push_dataset_name'], token=config['login_token'])

print("Dataset successfully uploaded to Hugging Face Hub.")
