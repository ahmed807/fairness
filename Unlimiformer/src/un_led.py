import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, LEDForConditionalGeneration,LEDTokenizer
from datasets import load_dataset
from unlimiformer import Unlimiformer
from usage import UnlimiformerArguments
import os
import logging
from huggingface_hub import login, HfApi
# Configure logging
logging.basicConfig(filename='Unlimiformer_LED.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
login(token='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')

# Set device to CPU for debugging
# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load the dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal")

tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')

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
# tokenized_test_dataset = dataset['test'].map(preprocess_function, batched=True, remove_columns=dataset['test'].column_names)



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


# Set generate hyperparameters
model.config.num_beams = 2
model.config.max_length = 1024
model.config.min_length = 256
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3


model.to(device)


# # Generate summaries
# def generate_summary(opinion):
#     inputs = tokenizer(opinion, return_tensors='pt', max_length=16384, truncation=True, padding='max_length')
#     summary_ids = model.generate(inputs['input_ids'].to("cuda"), max_length=1024, num_beams=4, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Use a data collator to handle padding
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer,
#     model=model,
#     padding=True,
#     max_length=16384
# )

# Define training arguments
training_args = TrainingArguments(
    output_dir='/srv/mostah/unlimiformer_results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=1,  # Reduce batch size for debugging
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='/srv/mostah/unlimiformer_results/logs',
    logging_steps=10,
    save_steps=1000,
    eval_steps=1000,
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset.select(range(20)),
    eval_dataset=tokenized_val_dataset.select(range(20)),
    # data_collator=data_collator
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
    # Evaluate the model
    logging.info("Evaluating the model")
    results = trainer.evaluate()
    logging.info(f"Evaluation results: {results}")

except RuntimeError as e:
    logging.error(f"Evaluation error occurred: {e}")
    print("Evaluation error details:")
    import traceback
    traceback.print_exc()



# Assuming `model` is your trained model
model.save_pretrained("led-base-16384/model")
tokenizer.save_pretrained("led-base-16384/model")

api = HfApi()

# Replace 'your-username/your-model-name' with your desired repository name
repo_id = "ahmed275/SS-LED_mixed_conservative_75"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
# Upload the model
api.upload_folder(
    folder_path="led-base-16384/model",
    repo_id=repo_id,
    repo_type="model"
)