import pandas as pd
import re
from datasets import Dataset, load_dataset,DatasetDict
from sklearn.model_selection import StratifiedShuffleSplit
from huggingface_hub import login, HfApi

# Replace 'your_huggingface_token' with your actual token
login(token='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')


dataset = load_dataset("ahmed275/OpinionOfTheCourt_dataset_new_updated")
df = pd.DataFrame(dataset['train'])

# Create a combined stratification column
df['stratify_col'] = df['issueArea'].astype(str) + '_' + df['decisionDirection'].astype(str)

# Filter out classes with fewer than 2 samples
class_counts = df['stratify_col'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['stratify_col'].isin(valid_classes)]

# Initialize StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, temp_index in split.split(df, df['stratify_col']):
    train_df = df.iloc[train_index]
    temp_df = df.iloc[temp_index]

# Filter out classes with fewer than 2 samples in the temporary dataset
temp_class_counts = temp_df['stratify_col'].value_counts()
valid_temp_classes = temp_class_counts[temp_class_counts >= 2].index
temp_df = temp_df[temp_df['stratify_col'].isin(valid_temp_classes)]

# Further split the temp_df into validation and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
for val_index, test_index in split.split(temp_df, temp_df['stratify_col']):
    val_df = temp_df.iloc[val_index]
    test_df = temp_df.iloc[test_index]

# Drop the stratification column
train_df = train_df.drop(columns=['stratify_col'])
val_df = val_df.drop(columns=['stratify_col'])
test_df = test_df.drop(columns=['stratify_col'])

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Print the distribution to verify
print("Training set distribution:")
print(train_df['issueArea'].value_counts(normalize=True))
print(train_df['decisionDirection'].value_counts(normalize=True))

print("\nValidation set distribution:")
print(val_df['issueArea'].value_counts(normalize=True))
print(val_df['decisionDirection'].value_counts(normalize=True))

print("\nTest set distribution:")
print(test_df['issueArea'].value_counts(normalize=True))
print(test_df['decisionDirection'].value_counts(normalize=True))

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub("ahmed275/Opinions_dataset_issueArea_decisionDirection_split")



# # Load the datasets
# dataset1 = load_dataset("ahmed275/OpinionOfTheCourt_dataset")
# dataset2 = load_dataset("ahmed275/cases_with_opinion")

# # Concatenate the datasets
# # Assuming both datasets have the same structure and are in the 'train' split
# combined_dataset = concatenate_datasets([dataset1['train'], dataset2['train']])

# # Create a DatasetDict if you want to maintain the 'train' split
# combined_dataset_dict = DatasetDict({'train': combined_dataset})

# # Save the combined dataset to a new dataset repository
# # Replace 'your-username/new-dataset-name' with your Hugging Face username and desired dataset name
# combined_dataset_dict.push_to_hub("ahmed275/OpinionOfTheCourt_dataset_new")