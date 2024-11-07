import yaml
import argparse
import torch._dynamo.config

# Add this line before creating the trainer
torch._dynamo.config.optimize_ddp = False
import logging


from datasets import load_dataset
from transformers import (
    LEDForConditionalGeneration,
    LEDTokenizer,
    LEDForConditionalGeneration,
    LongT5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from huggingface_hub import login, HfApi
import os 
# # Parse command-line arguments
# parser = argparse.ArgumentParser(description='Run LED model training.')
# parser.add_argument('config_path', type=str, help='Path to the configuration file')
# args = parser.parse_args()

# Load configuration
config_path = os.path.abspath('/root/fairness/config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)['led']

# Configure logging
logging.basicConfig(filename=config['log_file_name'], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Login to Hugging Face
login(token=config['login_token'])

# Load the dataset
dataset = load_dataset(config['dataset_name'])

# Load the tokenizer and model using eval
model= LongT5ForConditionalGeneration.from_pretrained('google/long-t5-tglobal-large')
tokenizer=AutoTokenizer.from_pretrained('google/long-t5-tglobal-large')

def preprocess_function(examples):
    logging.info(f"Preprocessing {len(examples)} examples")
    inputs = examples['opinionOfTheCourt']
    targets = examples['syllabus']
    model_inputs = tokenizer(inputs, max_length=16384, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=1024, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']

    batch = {}
    batch["input_ids"] = model_inputs.input_ids
    batch["attention_mask"] = model_inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = labels.input_ids

    # We have to make sure that the PAD token is ignored
    # -100 for loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

# Apply the preprocessing function to the datasets
logging.info("Tokenizing datasets")
tokenized_train_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
tokenized_val_dataset = dataset['validation'].map(preprocess_function, batched=True, remove_columns=dataset['validation'].column_names)

# logging.info(tokenized_train_dataset[0])
logging.info(len(tokenized_train_dataset[0]["input_ids"]))
# Set generate hyperparameters
model.config.num_beams = 4
model.config.max_length = 1024
model.config.min_length = 256
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./srv/results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,
    bf16 = True,
    torch_compile = True,
    optim = "adamw_torch_fused",
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# print(model)
# logging.info(model)

# Train the model
logging.info("Starting training")
trainer.train()

# Evaluate the model
logging.info("Evaluating the model")
results = trainer.evaluate()
logging.info(f"Evaluation results: {results}")

# Save the model
model.save_pretrained(config['pretrained_model_save_path'])
tokenizer.save_pretrained(config['pretrained_model_save_path'])

api = HfApi()

# Create and upload the model to the repository
api.create_repo(repo_id=config['repo_id'], repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=config['pretrained_model_save_path'],
    repo_id=config['repo_id'],
    repo_type="model"
)
