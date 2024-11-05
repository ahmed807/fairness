import logging
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import StratifiedShuffleSplit
from huggingface_hub import login
import os
from huggingface_hub import login
from huggingface_hub import HfApi


# Conservative (1) => 0
# Liberal      (2) => 1


# Configure logging
logging.basicConfig(filename='BERT-model_log_with_generated.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and list available devices
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("CUDA is available")
    print("Number of GPUs:", num_gpus)
    for i in range(num_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Set CUDA_VISIBLE_DEVICES to use GPU 0 if it exists
    if num_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")
        print("Using GPU 0")
else:
    print("CUDA is not available")

# Replace 'your_huggingface_token' with your actual token
login(token='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')

# Load the dataset from the Hugging Face Hub

dataset = load_dataset("ahmed275/Cases_with_LED_generated_summary")



# Check if the dataset is loaded correctly
if not dataset or 'train' not in dataset:
    logging.error("Failed to load dataset or 'train' split is missing.")
    exit()

# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

# Check if the DataFrame is loaded correctly
if df.empty:
    logging.error("DataFrame is empty after loading dataset.")
    exit()

logging.info("DataFrame shape: %s", df.shape)
logging.info("DataFrame head: %s", df.head())

# Initialize the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the data: set input as 'syllabus' and label as 'decisionDirection'
def preprocess_data(examples):
    inputs = bert_tokenizer(
        examples['syllabus'], 
        padding=True,
        truncation=True,
        max_length=512
    )
    # Adjust labels from 1, 2 to 0, 1
    labels = examples['decisionDirection']
    labels = [int(label) - 1 for label in labels]  # Adjust labels to start from 0
    num_labels = 2  # Assuming binary classification
    one_hot_labels = np.zeros((len(labels), num_labels))
    
    # Check if labels are within the expected range
    for i, label in enumerate(labels):
        if label < 0 or label >= num_labels:
            raise ValueError(f"Label {label} at index {i} is out of range for {num_labels} classes.")
    
    one_hot_labels[np.arange(len(labels)), labels] = 1
    inputs['labels'] = one_hot_labels.tolist()
    return inputs


# Apply the preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns(['id', 'year', 'url', 'opinionOfTheCourt', 'syllabus', 'issueArea', 'decisionDirection','generated_summary'])

# Check if the tokenized dataset is correct
if not tokenized_dataset or 'train' not in tokenized_dataset:
    logging.error("Tokenized dataset is empty or 'train' split is missing.")
    exit()

logging.info("Tokenized dataset size: %d", len(tokenized_dataset['train']))
logging.info("Sample tokenized data: %s", tokenized_dataset['train'][0])

# Split the dataset into train, validation, and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, temp_index in split.split(df, df['decisionDirection']):
    train_df = df.iloc[train_index]
    temp_df = df.iloc[temp_index]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in split.split(temp_df, temp_df['decisionDirection']):
    val_df = temp_df.iloc[val_index]
    test_df = temp_df.iloc[test_index]

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Ensure the datasets are tokenized
train_dataset = train_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

# Remove unnecessary columns from each dataset
train_dataset = train_dataset.remove_columns(['id', 'year', 'url', 'opinionOfTheCourt', 'syllabus', 'issueArea', 'decisionDirection','generated_summary'])
val_dataset = val_dataset.remove_columns(['id', 'year', 'url', 'opinionOfTheCourt', 'syllabus', 'issueArea', 'decisionDirection','generated_summary'])
test_dataset = test_dataset.remove_columns(['id', 'year', 'url', 'opinionOfTheCourt', 'syllabus', 'issueArea', 'decisionDirection','generated_summary'])

# Check if the datasets are not empty
if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
    logging.error("One of the datasets (train, validation, test) is empty.")
    exit()

logging.info("Train dataset size: %d", len(train_dataset))
logging.info("Validation dataset size: %d", len(val_dataset))
logging.info("Test dataset size: %d", len(test_dataset))

# Load the BERT model for binary classification
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Define training arguments for BERT
bert_training_args = TrainingArguments(
    output_dir='./bert_results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Use a data collator to handle padding
data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# Initialize the Trainer for BERT
bert_trainer = Trainer(
    model=bert_model,
    args=bert_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Train the BERT model
try:
    bert_trainer.train()
except Exception as e:
    logging.error("Error during training: %s", e)
    exit()

# Evaluate the BERT model
try:
    bert_results = bert_trainer.evaluate()
    logging.info("BERT Evaluation results: %s", bert_results)
except Exception as e:
    logging.error("Error during evaluation: %s", e)



# Predict decisionDirection using BERT with original syllabus
try:
    predictions_original = bert_trainer.predict(test_dataset)
    predicted_labels_original = np.argmax(predictions_original.predictions, axis=1)
    logging.info("Predictions (Original): %s", predictions_original.predictions)
    logging.info("Predicted Labels (Original): %s", predicted_labels_original)
except Exception as e:
    logging.error("Error during prediction with original syllabus: %s", e)


#####evaluation metrics (accuracy, f1 score) ################


# Preprocess the data: set input as 'generated_summary' and label as 'decisionDirection'
def preprocess_data_generated(examples):
    inputs = bert_tokenizer(
        examples['generated_summary'], 
        padding=True,
        truncation=True,
        max_length=512
    )
    # Adjust labels from 1, 2 to 0, 1
    labels = examples['decisionDirection']
    labels = [int(label) - 1 for label in labels]  # Adjust labels to start from 0
    num_labels = 2  # Assuming binary classification
    one_hot_labels = np.zeros((len(labels), num_labels))
    
    # Check if labels are within the expected range
    for i, label in enumerate(labels):
        if label < 0 or label >= num_labels:
            raise ValueError(f"Label {label} at index {i} is out of range for {num_labels} classes.")
    
    one_hot_labels[np.arange(len(labels)), labels] = 1
    inputs['labels'] = one_hot_labels.tolist()
    return inputs


# Prepare the test set using the generated summaries
test_df.loc[:, 'syllabus'] = df['generated_summary']

# Convert to Hugging Face Dataset
test_dataset_generated = Dataset.from_pandas(test_df)

# Tokenize the dataset using the generated summaries
test_dataset_generated = test_dataset_generated.map(preprocess_data_generated, batched=True)

# Remove unnecessary columns
test_dataset_generated = test_dataset_generated.remove_columns(['id', 'year', 'url', 'opinionOfTheCourt', 'syllabus', 'issueArea', 'decisionDirection', 'generated_summary'])

# Check if the test dataset is prepared correctly
logging.info("Test dataset size (generated): %d", len(test_dataset_generated))
logging.info("Sample test data (generated): %s", test_dataset_generated[0])



# Predict decisionDirection using BERT with generated summaries
try:
    predictions_generated = bert_trainer.predict(test_dataset_generated)
    predicted_labels_generated = np.argmax(predictions_generated.predictions, axis=1)
    logging.info("Predictions (Generated): %s", predictions_generated.predictions)
    logging.info("Predicted Labels (Generated): %s", predicted_labels_generated)
except Exception as e:
    logging.error("Error during prediction with generated summaries: %s", e)



# Calculate the distance-based metric between predictions
# try:
#     distance_metric = np.mean(predicted_labels_generated != predicted_labels_original)
#     logging.info("Distance-Based Metric between generated and original predictions: %f", distance_metric)
# except Exception as e:
#     logging.error("Error calculating distance-based metric: %s", e)


# Calculate the distance-based metric between predictions
try:
    # Calculate absolute differences
    absolute_differences = np.abs(predicted_labels_generated - predicted_labels_original)
    
    # Calculate the mean of absolute differences
    distance_metric = np.mean(absolute_differences)
    
    logging.info("Distance-Based Metric between generated and original predictions: %f", distance_metric)
except Exception as e:
    logging.error("Error calculating distance-based metric: %s", e)



true_labels = test_df['decisionDirection'].values - 1

logging.info("true_labels: %s", true_labels)
# Calculate the flip rate for predicted_labels_original
# Calculate the flip rate for predicted_labels_original
try:
    flips_0_to_1_original = np.sum((true_labels == 0) & (predicted_labels_original == 1))
    flips_1_to_0_original = np.sum((true_labels == 1) & (predicted_labels_original == 0))
    total_flips_original = flips_0_to_1_original + flips_1_to_0_original
    flip_rate_original = total_flips_original / len(true_labels)
    logging.info("Flip Rate for Original Predictions: %f", flip_rate_original)
    logging.info("Flips from 0 to 1 (Original): %d", flips_0_to_1_original)
    logging.info("Flips from 1 to 0 (Original): %d", flips_1_to_0_original)
except Exception as e:
    logging.error("Error calculating flip rate for original predictions: %s", e)

# Calculate the flip rate for predicted_labels_generated
try:
    flips_0_to_1_generated = np.sum((true_labels == 0) & (predicted_labels_generated == 1))
    flips_1_to_0_generated = np.sum((true_labels == 1) & (predicted_labels_generated == 0))
    total_flips_generated = flips_0_to_1_generated + flips_1_to_0_generated
    flip_rate_generated = total_flips_generated / len(true_labels)
    logging.info("Flip Rate for Generated Predictions: %f", flip_rate_generated)
    logging.info("Flips from 0 to 1 (Generated): %d", flips_0_to_1_generated)
    logging.info("Flips from 1 to 0 (Generated): %d", flips_1_to_0_generated)
except Exception as e:
    logging.error("Error calculating flip rate for generated predictions: %s", e)


### flip between original and generated


# # Step 1: Save the model and tokenizer locally
# model_dir = "bert_model_results"
# bert_model.save_pretrained(model_dir)
# bert_tokenizer.save_pretrained(model_dir)

# api = HfApi()

# # Replace 'your-username/your-model-name' with your desired repository name
# repo_id = "ahmed275/SS-BERT"
# api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
# # Upload the model
# api.upload_folder(
#     folder_path=model_dir,
#     repo_id=repo_id,
#     repo_type="model"
# )


# # Check CUDA availability
# logging.info("CUDA available: %s", torch.cuda.is_available())
# if torch.cuda.is_available():
#     logging.info("Current device: %d", torch.cuda.current_device())
#     logging.info("Device name: %s", torch.cuda.get_device_name(torch.cuda.current_device()))

