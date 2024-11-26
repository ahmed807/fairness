from transformers import BartForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments,DataCollatorWithPadding
from datasets import load_dataset
import torch
from unlimiformer import Unlimiformer
from usage import UnlimiformerArguments, training_addin
import os

# 
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define the preprocessing function
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

# Load the dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal")

# Initialize the tokenizer and model
modelname = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = BartForConditionalGeneration.from_pretrained(modelname)
model.resize_token_embeddings(len(tokenizer))

# Preprocess the dataset
tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
# Use a data collator to handle padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define Unlimiformer arguments
defaults = UnlimiformerArguments()
unlimiformer_kwargs = {
    'layer_begin': defaults.layer_begin,
    'layer_end': defaults.layer_end,
    'unlimiformer_head_num': defaults.unlimiformer_head_num,
    'exclude_attention': defaults.unlimiformer_exclude,
    'chunk_overlap': defaults.unlimiformer_chunk_overlap,
    'model_encoder_max_len': defaults.unlimiformer_chunk_size,
    'verbose': defaults.unlimiformer_verbose,
    'tokenizer': tokenizer,
    'unlimiformer_training': defaults.unlimiformer_training,
    'use_datastore': defaults.use_datastore,
    'flat_index': defaults.flat_index,
    'test_datastore': defaults.test_datastore,
    'reconstruct_embeddings': defaults.reconstruct_embeddings,
    'gpu_datastore': defaults.gpu_datastore,
    'gpu_index': defaults.gpu_index
}

# Convert the model to use Unlimiformer
model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
model.to(device)
# Define training arguments
training_args = TrainingArguments(
    output_dir='/srv/mostah/unlimiformer_results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=1000,
    eval_steps=1000,
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].select(range(20)),
    eval_dataset=tokenized_datasets['validation'].select(range(20)),
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
