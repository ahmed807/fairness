from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset,load_dataset
import numpy as np
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logging.basicConfig(filename='Longformer_syllabus_TEST_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal_test_generated_summaries")

logging.info("DATASET -> opinions_dataset_temporal_test_generated_summaries")

# Initialize the Longformer tokenizer
longformer_tokenizer = LongformerTokenizer.from_pretrained('ahmed275/SS-Longformer_syllabus_1')
# Load the Longformer model
longformer_model = LongformerForSequenceClassification.from_pretrained('ahmed275/SS-Longformer_syllabus_1', num_labels=2)

# Preprocess the data for 'syllabus'
def preprocess_data_syllabus(examples):
    # Tokenize the entire text
    inputs = longformer_tokenizer(
        examples['syllabus'],
        truncation=False,
        return_tensors=None  # Return python lists instead of tensors
    )
    # Take the last 4096 tokens
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Ensure the sequence is not longer than 4096 tokens
    if len(input_ids) > 4096:
        input_ids = input_ids[-4096:]
        attention_mask = attention_mask[-4096:]
    else:
        # Calculate padding length
        padding_length = 4096 - len(input_ids)
        
        # Pad input_ids with padding token
        padding_token = longformer_tokenizer.pad_token_id
        input_ids = input_ids + [padding_token] * padding_length
        
        # Pad attention mask with zeros
        attention_mask = attention_mask + [0] * padding_length

    assert len(input_ids) == 4096, f"Input ids length is {len(input_ids)}, should be 4096"
    assert len(attention_mask) == 4096, f"Attention mask length is {len(attention_mask)}, should be 4096"
    labels = int(examples['decisionDirection']) - 1
    assert labels in [0, 1], "Labels must be 0 or 1"
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Preprocess the data for 'opinionOfTheCourt'
def preprocess_data_opinion(examples):
    # Tokenize the entire text
    inputs = longformer_tokenizer(
        examples['opinionOfTheCourt'],
        truncation=False,
        return_tensors=None  # Return python lists instead of tensors
    )
    # Take the last 4096 tokens
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Ensure the sequence is not longer than 4096 tokens
    if len(input_ids) > 4096:
        input_ids = input_ids[-4096:]
        attention_mask = attention_mask[-4096:]
    else:
        # Calculate padding length
        padding_length = 4096 - len(input_ids)
        
        # Pad input_ids with padding token
        padding_token = longformer_tokenizer.pad_token_id
        input_ids = input_ids + [padding_token] * padding_length
        
        # Pad attention mask with zeros
        attention_mask = attention_mask + [0] * padding_length

    assert len(input_ids) == 4096, f"Input ids length is {len(input_ids)}, should be 4096"
    assert len(attention_mask) == 4096, f"Attention mask length is {len(attention_mask)}, should be 4096"
    labels = int(examples['decisionDirection']) - 1
    assert labels in [0, 1], "Labels must be 0 or 1"
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Tokenize the datasets
test_dataset_syllabus = dataset['train'].map(preprocess_data_syllabus, batched=False)
test_dataset_opinion = dataset['train'].map(preprocess_data_opinion, batched=False)


true_labels = [int(label) - 1 for label in dataset['train']['decisionDirection'][:10]]

# Initialize the Trainer
trainer = Trainer(
    model=longformer_model,
    args=TrainingArguments(
        output_dir='/srv/mostah/longformer_results',
        per_device_eval_batch_size=1    )
)

# Predict using the syllabus
predictions_syllabus = trainer.predict(test_dataset_syllabus.select(range(10)))
predicted_labels_syllabus = np.argmax(predictions_syllabus.predictions, axis=1)

# Predict using the opinion of the court
predictions_opinion = trainer.predict(test_dataset_opinion.select(range(10)))
predicted_labels_opinion = np.argmax(predictions_opinion.predictions, axis=1)

logging.info("Predictions for Syllabus: %s", np.array2string(predictions_syllabus.predictions, separator=', '))

# Log the predictions array for opinion
logging.info("Predictions for Opinion: %s", np.array2string(predictions_opinion.predictions, separator=', '))


for i in range(10):
    true_label = true_labels[i]
    predicted_label_syllabus = predicted_labels_syllabus[i]
    predicted_label_opinion = predicted_labels_opinion[i]
    logging.info(" True Label: %d, Predicted Syllabus: %d, Predicted Opinion: %d", true_label, predicted_label_syllabus, predicted_label_opinion)

accuracy = accuracy_score(true_labels, predicted_labels_syllabus)
precision = precision_score(true_labels, predicted_labels_syllabus, average='binary')
recall = recall_score(true_labels, predicted_labels_syllabus, average='binary')
f1 = f1_score(true_labels, predicted_labels_syllabus, average='binary')
logging.info("Syllabus Accuracy: %f", accuracy)
logging.info("Syllabus Precision: %f", precision)
logging.info("Syllabus Recall: %f", recall)
logging.info("Syllabus F1 Score: %f", f1)

accuracy = accuracy_score(true_labels, predicted_labels_opinion)
precision = precision_score(true_labels, predicted_labels_opinion, average='binary')
recall = recall_score(true_labels, predicted_labels_opinion, average='binary')
f1 = f1_score(true_labels, predicted_labels_opinion, average='binary')
logging.info("Opinion Accuracy: %f", accuracy)
logging.info("Opinion Precision: %f", precision)
logging.info("Opinion Recall: %f", recall)
logging.info("Opinion F1 Score: %f", f1)


# Calculate absolute differences
absolute_differences = np.abs(predicted_labels_syllabus - predicted_labels_opinion)
# Calculate the mean of absolute differences
distance_metric_absolute_differences = np.mean(absolute_differences)
logging.info("Distance-Based Metric absolute_differences: %f", distance_metric_absolute_differences)

# # Calculate the distance-based metric
# distance_metric = np.mean(predicted_labels_syllabus != predicted_labels_opinion)
# logging.info("Distance-Based Metric: %f", distance_metric)

# Calculate the flip rate for syllabus predictions
flips_syllabus = np.sum(predicted_labels_syllabus != true_labels)
flip_rate_syllabus = flips_syllabus / len(true_labels)
logging.info("Flip Rate for Syllabus Predictions: %f", flip_rate_syllabus)

# Calculate the flip rate for opinion predictions
flips_opinion = np.sum(predicted_labels_opinion != true_labels)
flip_rate_opinion = flips_opinion / len(true_labels)
logging.info("Flip Rate for Opinion Predictions: %f", flip_rate_opinion)


# Calculate the macro F1 score for syllabus predictions
macro_f1_syllabus = f1_score(true_labels, predicted_labels_syllabus, average='macro')
logging.info(f"Macro F1 Score for Syllabus: {macro_f1_syllabus}")

# Calculate the macro F1 score for opinion predictions
macro_f1_opinion = f1_score(true_labels, predicted_labels_opinion, average='macro')
logging.info(f"Macro F1 Score for Opinion: {macro_f1_opinion}")
