import json
import logging
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from huggingface_hub import login, HfApi
import sled
import torch

# Configure logging
logging.basicConfig(filename='BART_SLED.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Login to Hugging Face
login(token='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')

# Load dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal")
df = pd.DataFrame(dataset['test'])

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('tau/bart-base-sled')
model = AutoModelForSeq2SeqLM.from_pretrained('tau/bart-base-sled')  # Changed to Seq2SeqLM

# Set maximum lengths
MAX_INPUT_LENGTH = 16384  # Adjust based on your GPU memory
MAX_TARGET_LENGTH = 1024

def preprocess_function(examples):
    logging.info(f"Preprocessing {len(examples)} examples")
    inputs = examples['opinionOfTheCourt']
    targets = examples['syllabus']
    
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

    model_inputs['labels'] = labels['input_ids']
    
    # Replace padding token id with -100 for loss calculation
    model_inputs['labels'] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in model_inputs['labels']
    ]

    return model_inputs

# Create data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    max_length=MAX_INPUT_LENGTH
)

# Tokenize datasets
logging.info("Tokenizing datasets")
tokenized_train_dataset = dataset['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Running tokenizer on train dataset"
)
tokenized_val_dataset = dataset['validation'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['validation'].column_names,
    desc="Running tokenizer on validation dataset"
)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='/srv/mostah/sled_results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,  # Disabled FP16
    # gradient_accumulation_steps=16,  # Added gradient accumulation
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,
    logging_steps=100,
    save_steps=1000,
    bf16 = True,
    gradient_checkpointing=True
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

# Train the model
logging.info("Starting training")
trainer.train()

# Generate summaries function
def generate_summary(opinion):
    inputs = tokenizer(
        opinion,
        return_tensors='pt',
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    ).to(model.device)
    
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        length_penalty=2.0
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Save and upload model
model.save_pretrained("bart-base-sled/model")
tokenizer.save_pretrained("bart-base-sled/model")

api = HfApi()
repo_id = "ahmed275/BART-SLED_temporal"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="bart-base-sled/model",
    repo_id=repo_id,
    repo_type="model"
)

# Test model
sample_opinion = df['opinionOfTheCourt'].iloc[0]
logging.info(f"Testing model on sample opinion: {sample_opinion}")
logging.info(f"Generated Summary: {generate_summary(sample_opinion)}")

# Generate summaries for all test cases
df['generated_summary'] = df['opinionOfTheCourt'].apply(generate_summary)
