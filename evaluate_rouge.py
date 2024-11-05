import logging
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
import os
import torch

# Configure logging
logging.basicConfig(
    filename='evaluation.log',  # Log output file
    level=logging.INFO,         # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Load the dataset
logging.info("Loading dataset from Hugging Face...")
dataset = load_dataset("ahmed275/Cases_with_LED_generated_summary")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['train'])
logging.info("Dataset loaded and converted to DataFrame.")

# Initialize the RougeScorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
logging.info("Initialized ROUGE scorer.")

# Group the data by 'issueArea'
grouped = df.groupby('issueArea')

# Prepare a dictionary to store average scores for each issue area
average_scores = {}

# Calculate ROUGE scores for each group
for issue_area, group in grouped:
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for index, row in group.iterrows():
        reference_summary = row['opinionOfTheCourt']
        candidate_summary = row['generated_summary']
        
        scores = scorer.score(reference_summary, candidate_summary)
        
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
        logging.debug(f"Issue Area {issue_area} - Processed row {index}: ROUGE-1: {scores['rouge1'].fmeasure}, ROUGE-2: {scores['rouge2'].fmeasure}, ROUGE-L: {scores['rougeL'].fmeasure}")
    
    # Calculate average ROUGE scores for this issue area
    average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    average_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    average_scores[issue_area] = {
        'rouge1': average_rouge1,
        'rouge2': average_rouge2,
        'rougeL': average_rougeL
    }
    
    logging.info(f'Issue Area {issue_area} - Average ROUGE-1 F1 Score: {average_rouge1:.4f}')
    logging.info(f'Issue Area {issue_area} - Average ROUGE-2 F1 Score: {average_rouge2:.4f}')
    logging.info(f'Issue Area {issue_area} - Average ROUGE-L F1 Score: {average_rougeL:.4f}')

# Print the average scores for each issue area
for issue_area, scores in average_scores.items():
    print(f'Issue Area {issue_area} - Average ROUGE-1 F1 Score: {scores["rouge1"]:.4f}')
    print(f'Issue Area {issue_area} - Average ROUGE-2 F1 Score: {scores["rouge2"]:.4f}')
    print(f'Issue Area {issue_area} - Average ROUGE-L F1 Score: {scores["rougeL"]:.4f}')
