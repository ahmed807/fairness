import logging
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
import os
# Configure logging
logging.basicConfig(
    filename='scores_conservative.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
# if torch.cuda.is_available():
#     num_gpus = torch.cuda.device_count()
#     print("CUDA is available")
#     print("Number of GPUs:", num_gpus)
#     for i in range(num_gpus):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
#     # Set CUDA_VISIBLE_DEVICES to use GPU 0 if it exists
#     if num_gpus > 1:
#         os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#         device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#         logging.info(f"Device: {device}")
# else:
#     print("CUDA is not available")
# Load the dataset
logging.info("Loading dataset from Hugging Face...")
print("Loading dataset from Hugging Face...")
dataset = load_dataset("ahmed275/opinions_dataset_temporal_test_generated_summaries")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['train'])
logging.info("Dataset loaded and converted to DataFrame.")
print("Dataset loaded and converted to DataFrame.")

# Initialize the RougeScorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
logging.info("Initialized ROUGE scorer.")
print("Initialized ROUGE scorer.")

# Initialize BERTScorer
bert_scorer = BERTScorer(model_type='bert-base-uncased')
logging.info("Initialized BERT scorer.")
print("Initialized BERT scorer.")

# Prepare a list to store scores for each case
scores_list = []

# Calculate ROUGE and BERT scores for each case
for index, row in df.iterrows():
    reference_summary = row['syllabus']
    candidate_summary = row['generated_summary']

    # Calculate ROUGE scores
    scores = scorer.score(reference_summary, candidate_summary)
    rouge1_score = scores['rouge1'].fmeasure
    rouge2_score = scores['rouge2'].fmeasure
    rougeL_score = scores['rougeL'].fmeasure

    # Calculate BERTScore
    P, R, F1 = bert_scorer.score([candidate_summary], [reference_summary])
    bert_f1_score = F1.mean().item()

    # Append scores to the list
    scores_list.append({
        'index': index,
        'issueArea': row['issueArea'],
        'decisionDirection': row['decisionDirection'],
        'partyWinning': row['partyWinning'],
        'voteDistribution': row['voteDistribution'],
        'respondentType': row['respondentType'],
        'rouge1': rouge1_score,
        'rouge2': rouge2_score,
        'rougeL': rougeL_score,
        'bert_f1': bert_f1_score
    })

    logging.debug(f"Processed row {index}: ROUGE-1: {rouge1_score}, ROUGE-2: {rouge2_score}, ROUGE-L: {rougeL_score}, BERT-F1: {bert_f1_score}")

# Convert the scores list to a DataFrame
scores_df = pd.DataFrame(scores_list)

# Print the DataFrame
print(scores_df)

# Optionally, save the DataFrame to a CSV file
scores_df.to_csv('case_scores_conservative.csv', index=False)
