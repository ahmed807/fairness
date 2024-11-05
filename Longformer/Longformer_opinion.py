from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import login, HfApi
import torch
import os

logging.basicConfig(filename='Longformer_opinion_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal")
login(token='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')

# Initialize the Longformer tokenizer
longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# Preprocess the data: set input as 'opinion' and label as 'decisionDirection'
def preprocess_data(examples):
    # Tokenize the entire text
    inputs = longformer_tokenizer(
        examples['opinionOftheCourt'],
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
    

    # print(len(input_ids))
    # Adjust labels from 1, 2 to 0, 1
    # labels = examples['decisionDirection']
    # labels = [int(label) - 1 for label in labels]  # Adjust labels to start from 0
    labels = int(examples['decisionDirection']) - 1
    assert labels in [0, 1], "Labels must be 0 or 1"
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# Apply the preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess_data, batched=False,remove_columns=dataset['train'].column_names)
tokenized_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

# Load the Longformer model for binary classification
longformer_model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)
longformer_model = torch.compile(longformer_model)
# .to(device)

# Define training arguments for Longformer
training_args = TrainingArguments(
    output_dir='/srv/mostah/longformer_results',
    # eval_strategy='epoch',
    eval_strategy='steps',
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust batch size for memory constraints
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir='/srv/mostah/logs',
    logging_steps=100,
    # Gradient Accumulation Settings
    # gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    fp16=True,
    remove_unused_columns=False,
    max_grad_norm=1.0,

    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',  # Monitor evaluation loss
    greater_is_better=False,  # Lower loss is better
    warmup_steps=500,  # Number of steps to warm up the learning rate
    lr_scheduler_type='linear',  # Type of learning rate scheduler


    
)
# Use a data collator that handles padding
data_collator = DataCollatorWithPadding(tokenizer=longformer_tokenizer)

# Initialize the Trainer for Longformer
trainer = Trainer(
    model=longformer_model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],#.select(range(20))
    eval_dataset=tokenized_dataset['validation'],#.select(range(10)),
    data_collator=data_collator
)

# Train the Longformer model
try:
    trainer.train()
except Exception as e:
    logging.error("Error during training: %s", e)

# Evaluate the Longformer model
try:
    results = trainer.evaluate()
    print(results)
    logging.info("Longformer Evaluation results: %s", results)
except Exception as e:
    logging.error("Error during evaluation: %s", e)

# Predict on the test dataset
test_dataset = tokenized_dataset['test']#.select(range(10))  # Ensure you have a test split
predictions = trainer.predict(test_dataset)


print("predictions:",predictions)
logging.info("Predictions type: %s", type(predictions.predictions))

# Extract logits from the predictions tuple
logits = predictions.predictions[1]  # Assuming the second element contains the logits

# Log the shape of logits
logging.info("Logits shape: %s", np.shape(logits))

# predicted_labels = np.argmax(predictions.predictions, axis=1)

# Use argmax to get predicted labels from logits
predicted_labels = np.argmax(logits, axis=1)

# Calculate metrics
true_labels = [example['labels'] for example in test_dataset]

for i in range(10):
    true_label = true_labels[i]
    predicted_label = predicted_labels[i]
    logging.info(" True Label: %d, Predicted Label: %d", true_label, predicted_label)


accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='binary')
recall = recall_score(true_labels, predicted_labels, average='binary')
f1 = f1_score(true_labels, predicted_labels, average='binary')

# Log the results
logging.info("Accuracy: %f", accuracy)
logging.info("Precision: %f", precision)
logging.info("Recall: %f", recall)
logging.info("F1 Score: %f", f1)

# Save the model and tokenizer locally
model_dir = "/srv/mostah/longformer_model_results"
longformer_model.save_pretrained(model_dir)
longformer_tokenizer.save_pretrained(model_dir)

# Upload the model to Hugging Face
api = HfApi()
repo_id = "ahmed275/SS-Longformer_opinion"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    repo_type="model"
)
