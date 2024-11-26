import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import json
import logging
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from huggingface_hub import login, HfApi
import torch
from unlimiformer import Unlimiformer
from usage import UnlimiformerArguments

# Configure logging
logging.basicConfig(filename='unlimiformer_training.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Login to Hugging Face
login(token='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')

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")

# Initialize device
device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal")
df = pd.DataFrame(dataset['test'])

def initialize_model(training=True):
    """Initialize model with appropriate settings for training or evaluation"""
    try:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        
        # # Enable gradient checkpointing if supported
        # if hasattr(base_model.config, 'gradient_checkpointing'):
        #     base_model.config.gradient_checkpointing = True
        
        # Set up Unlimiformer arguments with only valid parameters
        defaults = UnlimiformerArguments()
        unlimiformer_kwargs = {
            'layer_begin': defaults.layer_begin,
            'layer_end': defaults.layer_end,
            'tokenizer': tokenizer,
            'model_encoder_max_len': defaults.unlimiformer_chunk_size,
            'chunk_overlap': defaults.unlimiformer_chunk_overlap,
            'verbose': defaults.unlimiformer_verbose,
            'use_datastore': defaults.use_datastore,
            'flat_index': defaults.flat_index,
            'test_datastore': defaults.test_datastore,
            'reconstruct_embeddings': defaults.reconstruct_embeddings,
            'gpu_datastore': defaults.gpu_datastore,
            'gpu_index': defaults.gpu_index
        }
        
        if training:
            unlimiformer_kwargs.update({
                'unlimiformer_training': True
            })
        
        # Convert and move model to device
        model = Unlimiformer.convert_model(base_model, **unlimiformer_kwargs)
        model = model.to(device)
        
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Error during model initialization: {str(e)}")
        raise

def preprocess_function(examples, tokenizer, max_length=16384):
    """Preprocess function with appropriate truncation"""
    inputs = examples['opinionOfTheCourt']
    targets = examples['syllabus']
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"  # Ensure tensors are returned
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors="pt"  # Ensure tensors are returned
        )

    # Convert labels to tensor and replace -100 for padding
    labels['input_ids'] = torch.tensor([
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels['input_ids']
    ])

    model_inputs['labels'] = labels['input_ids']
    
    # Ensure all model inputs are tensors and move them to the device
    return {k: v.to(device) for k, v in model_inputs.items()}
# Initialize model and tokenizer
model, tokenizer = initialize_model(training=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='/srv/mostah/unlimiformer_results',
    eval_strategy='epoch',  # Use 'epoch' or 'steps' for both
    save_strategy='epoch',  # Align with eval_strategy
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
    gradient_accumulation_steps=16,
    generation_max_length=1024,
    generation_num_beams=4,
    logging_steps=100,
    save_steps=1000,  # This will be ignored if save_strategy is 'epoch'
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    ),
    eval_dataset=dataset['validation'].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset['validation'].column_names
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# Train the model
trainer.train()

# Save the model
save_directory = "/srv/mostah/unlimiformer_model"
os.makedirs(save_directory, exist_ok=True)

try:
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # Upload to HuggingFace
    api = HfApi()
    repo_id = "ahmed275/unlimiformer_temporal"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=save_directory,
        repo_id=repo_id,
        repo_type="model"
    )
    logging.info(f"Model successfully saved and uploaded to {repo_id}")
    
except Exception as e:
    logging.error(f"Error during save or upload: {str(e)}")
