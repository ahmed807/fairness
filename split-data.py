from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import random
from huggingface_hub import login
from huggingface_hub import HfApi


login(token='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')





# Load the dataset from the Hugging Face Hub
dataset = load_dataset("ahmed275/opinions_dataset_temporal")

# Convert the training dataset to a pandas DataFrame for easier manipulation
train_df = pd.DataFrame(dataset['train'])

if '__index_level_0__' in train_df.columns:
    train_df = train_df.drop(columns=['__index_level_0__'])

# Filter training datasets based on decisionDirection
conservative_train_df = train_df[train_df['decisionDirection'] == 1]
liberal_train_df = train_df[train_df['decisionDirection'] == 2]

# Create a mixed training dataset with 50% from each
conservative_train_sample = conservative_train_df.sample(frac=0.25, random_state=42)
liberal_train_sample = liberal_train_df.sample(frac=0.75, random_state=42)
mixed_train_df = pd.concat([conservative_train_sample, liberal_train_sample])

# Convert DataFrames back to Dataset objects for training
conservative_train_dataset = Dataset.from_pandas(conservative_train_df)
liberal_train_dataset = Dataset.from_pandas(liberal_train_df)
mixed_train_dataset = Dataset.from_pandas(mixed_train_df)

# Convert the validation dataset to a pandas DataFrame for easier manipulation
validation_df = pd.DataFrame(dataset['validation'])

if '__index_level_0__' in validation_df.columns:
    validation_df = validation_df.drop(columns=['__index_level_0__'])

# Filter validation datasets based on decisionDirection
conservative_validation_df = validation_df[validation_df['decisionDirection'] == 1]
liberal_validation_df = validation_df[validation_df['decisionDirection'] == 2]

# Create a mixed validation dataset with 50% from each
conservative_validation_sample = conservative_validation_df.sample(frac=0.25, random_state=42)
liberal_validation_sample = liberal_validation_df.sample(frac=0.75, random_state=42)
mixed_validation_df = pd.concat([conservative_validation_sample, liberal_validation_sample])

# Convert DataFrames back to Dataset objects for validation
conservative_validation_dataset = Dataset.from_pandas(conservative_validation_df)
liberal_validation_dataset = Dataset.from_pandas(liberal_validation_df)
mixed_validation_dataset = Dataset.from_pandas(mixed_validation_df)

# # Create DatasetDicts for each type
# conservative_dataset_temporal = DatasetDict({
#     'train': conservative_train_dataset,
#     'validation': conservative_validation_dataset
# })

# liberal_dataset_temporal = DatasetDict({
#     'train': liberal_train_dataset,
#     'validation': liberal_validation_dataset
# })

mixed_dataset_temporal = DatasetDict({
    'train': mixed_train_dataset,
    'validation': mixed_validation_dataset
})

# # Push the datasets to the Hugging Face Hub
# conservative_dataset_temporal.push_to_hub("ahmed275/conservative_dataset_temporal", private=False)
# liberal_dataset_temporal.push_to_hub("ahmed275/liberal_dataset_temporal", private=False)
mixed_dataset_temporal.push_to_hub("ahmed275/mixed_dataset_temporal_liberal_75", private=False)
