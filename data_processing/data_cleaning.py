import re
import pandas as pd
import logging
from datasets import load_dataset, Dataset



logging.basicConfig(
    filename='data_cleaning.log',  # Log output file
    level=logging.INFO,         # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Load the dataset
logging.info("Loading dataset from Hugging Face Hub.")
dataset = load_dataset("ahmed275/cases_with_opinion")

# pattern_page = r'(?i)(?:Pp\.|Page)?\s*\d+\s*U\. S\.\s*\d+(?:-\d+)?'
# pattern_syllabus = r'.*?Syllabus\s*'
pattern_pp = r'(?<!App\.)Pp\. \d+–\d+'
# Define the regex patterns
pattern_footnote = r'\[Footnote \d+\]'


# Function to clean the opinionOfTheCourt field
def clean_opinion(text):
    # Remove page references
    # text = re.sub(pattern_page, '', text)
    # Remove footnote references
    text = re.sub(pattern_footnote, '', text)
    return text
# Function to clean the syllabus field
def clean_syllabus(text):
    # Remove everything before and including "Syllabus"
    # text = re.sub(pattern_syllabus, '', text)
    # Remove specific patterns
    return re.sub(pattern_pp, '', text)
def clean_text(text):
    # Remove all newline characters
    text = text.replace('\n', ' ')
    
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    
    return text
# Clean the dataset
logging.info("Cleaning the dataset.")
cleaned_data = []
for item in dataset['train']:
    try:
        cleaned_item = {
            'id': item['id'],
            'year': item['year'],
            'url': item['url'],
            'opinionOfTheCourt': clean_syllabus(item['opinionOfTheCourt']),
            'syllabus': clean_syllabus(item['syllabus']),
            'issueArea': item['issueArea'],
            'decisionDirection': item['decisionDirection']
        }
        cleaned_data.append(cleaned_item)
    except Exception as e:
        logging.error(f"Error cleaning item with id {item['id']}: {e}")

# Convert cleaned data to a Pandas DataFrame
logging.info("Converting cleaned data to Pandas DataFrame.")
df = pd.DataFrame(cleaned_data)

# Convert DataFrame to a Dataset object
logging.info("Converting DataFrame to Dataset object.")
cleaned_dataset = Dataset.from_pandas(df)

# Push the dataset to the Hugging Face Hub
name = "cases_with_opinion"  # Replace with your desired dataset name
logging.info(f"Pushing dataset to Hugging Face Hub with name: {name}.")
try:
    cleaned_dataset.push_to_hub(f"ahmed275/{name}", private=False)
    logging.info("Dataset successfully pushed to Hugging Face Hub.")
except Exception as e:
    logging.error(f"Failed to push dataset to Hugging Face Hub: {e}")
