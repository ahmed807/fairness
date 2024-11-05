


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
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from datasets import load_dataset, Dataset
import numpy as np
import logging
import os

# Set to use GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)



def set_gpu_device(gpu_number):
    # Check if CUDA is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("CUDA is available")
        print("Number of GPUs:", num_gpus)
        
        # List available devices
        for i in range(num_gpus):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Check if the specified GPU number is valid
        if 0 <= gpu_number < num_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
            device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
            logging.info(f"Device: {device}")
            print(f"Using GPU {gpu_number}")
        else:
            print(f"Invalid GPU number: {gpu_number}. Please choose a number between 0 and {num_gpus - 1}.")
    else:
        print("CUDA is not available")

# Example usage
set_gpu_device(0)

# Configure logging
logging.basicConfig(filename='test_BERT.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace 'your_huggingface_token' with your actual token
login(token='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')




# Load the model and tokenizer from the Hugging Face Hub
model_dir = "ahmed275/SS-BERT"
bert_model = BertForSequenceClassification.from_pretrained(model_dir)
bert_tokenizer = BertTokenizer.from_pretrained(model_dir)

# Load the dataset
dataset = load_dataset("ahmed275/Cases_with_LED_generated_summary")

# Prepare the original test set
# Assuming 'test_dataset' is already prepared and available
# If not, you need to prepare it similarly to how 'test_dataset_generated' is prepared

# Initialize the Trainer with the loaded model
bert_trainer = Trainer(
    model=bert_model,
    tokenizer=bert_tokenizer
)

# Predict decisionDirection using BERT with original syllabus
try:
    predictions_original = bert_trainer.predict(test_dataset)
    predicted_labels_original = np.argmax(predictions_original.predictions, axis=1)
except Exception as e:
    logging.error("Error during prediction with original syllabus: %s", e)

# Prepare the test set using the generated summaries
test_df = dataset['train'].to_pandas()  # Assuming the dataset has a 'train' split
test_df['syllabus'] = test_df['generated_summary']

# Convert to Hugging Face Dataset
test_dataset_generated = Dataset.from_pandas(test_df)

# Check if the test dataset is prepared correctly
logging.info("Test dataset size (generated): %d", len(test_dataset_generated))
logging.info("Sample test data (generated): %s", test_dataset_generated[0])

# Predict decisionDirection using BERT with generated summaries
try:
    predictions_generated = bert_trainer.predict(test_dataset_generated)
    predicted_labels_generated = np.argmax(predictions_generated.predictions, axis=1)
except Exception as e:
    logging.error("Error during prediction with generated summaries: %s", e)

# Calculate the distance-based metric between predictions
try:
    distance_metric = np.mean(predicted_labels_generated != predicted_labels_original)
    logging.info("Distance-Based Metric between generated and original predictions: %f", distance_metric)
except Exception as e:
    logging.error("Error calculating distance-based metric: %s", e)
