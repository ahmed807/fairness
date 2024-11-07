import json
import logging
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Trainer
)
from huggingface_hub import login, HfApi
import sled  # ** required so SLED would be properly registered by the AutoClasses **
from transformers import AutoTokenizer, AutoModel
import torch
# Configure logging
import torch


logging.basicConfig(filename='BART_SLED.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace 'your_huggingface_token' with your actual token
login(token='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')

dataset = load_dataset("ahmed275/opinions_dataset_temporal")
df = pd.DataFrame(dataset['test'])

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('tau/bart-base-sled')
model = AutoModel.from_pretrained('tau/bart-base-sled')


def preprocess_function(examples):
    logging.info(f"Preprocessing {len(examples)} examples")
    inputs = examples['opinionOfTheCourt']
    targets = examples['syllabus']
    # model_inputs = tokenizer(inputs, max_length=16384, truncation=True, padding='max_length')
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    model_inputs['labels'] = labels['input_ids']

    # We have to make sure that the PAD token is ignored
    # -100 for loss
    # model_inputs["labels"] = [
    #     [-100 if token == tokenizer.pad_token_id else token for token in labels]
    #     for labels in model_inputs["labels"]
    # ]

    return model_inputs

# Apply the preprocessing function to the datasets
logging.info("Tokenizing datasets")
tokenized_train_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
tokenized_val_dataset = dataset['validation'].map(preprocess_function, batched=True, remove_columns=dataset['validation'].column_names)


model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
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

# Generate summaries
def generate_summary(opinion):
    inputs = tokenizer(opinion, return_tensors='pt', max_length=16384, truncation=True, padding='max_length')
    summary_ids = model.generate(inputs['input_ids'].to("cuda"), max_length=1024, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Save the model
model.save_pretrained("bart-base-sled/model")
tokenizer.save_pretrained("bart-base-sled/model")

api = HfApi()

# Replace 'your-username/your-model-name' with your desired repository name
repo_id = "ahmed275/BART-SLED_temporal"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
# Upload the model
api.upload_folder(
    folder_path="bart-base-sled/model",
    repo_id=repo_id,
    repo_type="model"
)

# Test the model on a sample opinion
sample_opinion = df['opinionOfTheCourt'].iloc[0]
logging.info(f"Testing model on sample opinion: {sample_opinion}")
logging.info(f"Generated Summary: {generate_summary(sample_opinion)}")

df['generated_summary'] = df['opinionOfTheCourt'].apply(generate_summary)
