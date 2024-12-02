import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from unlimiformer import Unlimiformer
from usage import UnlimiformerArguments
import os
import logging

# Configure logging
logging.basicConfig(filename='Unlimiformer_BART.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device to CPU for debugging
# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load the dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal")

# Initialize the tokenizer and model
modelname = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForSeq2SeqLM.from_pretrained(modelname)


# tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
# model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')

# Set maximum lengths
MAX_INPUT_LENGTH = 16384  # Adjust based on your GPU memory
MAX_TARGET_LENGTH = 1024

# Define the preprocessing function
def preprocess_function(examples):
    logging.info(f"Preprocessing {len(examples)} examples")
    inputs = examples['opinionOfTheCourt']
    targets = examples['syllabus']
    
    model_inputs = tokenizer(
        inputs,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
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

# Preprocess the dataset
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
model.resize_token_embeddings(len(tokenizer))

# Define Unlimiformer arguments
defaults = UnlimiformerArguments()
unlimiformer_kwargs = {
    'layer_begin': defaults.layer_begin,
    'layer_end': defaults.layer_end,
    'unlimiformer_head_num': defaults.unlimiformer_head_num,
    'exclude_attention': defaults.unlimiformer_exclude,
    'chunk_overlap': defaults.unlimiformer_chunk_overlap,
    'model_encoder_max_len': 1000,
    'verbose': defaults.unlimiformer_verbose,
    'tokenizer': tokenizer,
    'unlimiformer_training': defaults.unlimiformer_training,
    'use_datastore': defaults.use_datastore,
    'flat_index': defaults.flat_index,
    'test_datastore': defaults.test_datastore,
    'reconstruct_embeddings': defaults.reconstruct_embeddings,
    'gpu_datastore': defaults.gpu_datastore,
    'gpu_index': defaults.gpu_index,
}

# Convert the model to use Unlimiformer
model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
model.to(device)

# Use a data collator to handle padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    max_length=MAX_INPUT_LENGTH
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/srv/mostah/unlimiformer_results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=1,  # Reduce batch size for debugging
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='/srv/mostah/unlimiformer_results/logs',
    logging_steps=10,
    save_steps=1000,
    eval_steps=1000,
    load_best_model_at_end=True,
    fp16=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset.select(range(5)),
    eval_dataset=tokenized_val_dataset.select(range(5)),
    data_collator=data_collator
)

# Train the model
try:
    trainer.train()
except RuntimeError as e:
    logging.error(f"Training error occurred: {e}")
    print("Error details:")
    import traceback
    traceback.print_exc()

# Evaluate the model
try:
    trainer.evaluate()
except RuntimeError as e:
    logging.error(f"Evaluation error occurred: {e}")
    print("Evaluation error details:")
    import traceback
    traceback.print_exc()
