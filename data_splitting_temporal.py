import pandas as pd
from datasets import load_dataset, Dataset,DatasetDict
from huggingface_hub import login, HfApi

# Load the dataset
dataset = load_dataset("ahmed275/OpinionOfTheCourt_dataset_new_updated")
df = pd.DataFrame(dataset['train'])

df['year'] = df['year'].astype(int)

train_df = df[(df['year'] >= 1955) & (df['year'] <= 2002)]
val_df = df[(df['year'] >= 2003) & (df['year'] <= 2010)]
test_df = df[(df['year'] >= 2011) & (df['year'] <= 2020)]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

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
dataset_dict.push_to_hub("ahmed275/opinions_dataset_temporal")
