import logging
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
import os
import json
from AlignScore.src.alignscore import AlignScore
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
import nltk
from tqdm import tqdm

nltk.download('punkt_tab')
# print("Using GPU ", os.environ["CUDA_VISIBLE_DEVICES"])

# Configure logging
logging.basicConfig(
    filename='./all_scores.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Load the JSON configuration file
with open('all_scores_config.json', 'r') as f:
    datasets_config = json.load(f)

# Initialize all scorers
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert_scorer = BERTScorer(model_type='bert-base-uncased')
align_scorer = AlignScore(
    model='roberta-base',
    batch_size=32,
    device='cuda:0',
    ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt',
    evaluation_mode='nli_sp'
)

# Initialize UniEval evaluator
task = 'summarization'
evaluator = get_evaluator(task)

logging.info("Initialized all scorers.")
print("Initialized all scorers.")

# Loop through each dataset configuration
for index, config in tqdm(enumerate(datasets_config), desc="Configs", total=len(datasets_config)):
    dataset_name = config['dataset_name']
    csv_filename = config['csv_filename']

    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # Load the dataset
    logging.info(f"Loading dataset {dataset_name} from Hugging Face...")
    print(f"Loading dataset {dataset_name} from Hugging Face...")
    dataset = load_dataset(dataset_name)

    logging.info(f"DATASET {dataset_name} all scorers.")
    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    logging.info("Dataset loaded and converted to DataFrame.")
    print("Dataset loaded and converted to DataFrame.")

    # Prepare a list to store scores for each case
    scores_list = []

    # Calculate scores for each case
    for index, row in tqdm(df.iterrows(), desc=f"Processing {dataset_name}", total=len(df)):
        reference_summary = row['syllabus']
        candidate_summary = row['generated_summary']
        opinion_of_the_court = row['opinionOfTheCourt']

        # Calculate ROUGE scores
        scores = scorer.score(reference_summary, candidate_summary)
        rouge1_score = scores['rouge1'].fmeasure
        rouge2_score = scores['rouge2'].fmeasure
        rougeL_score = scores['rougeL'].fmeasure

        # Calculate BERTScore
        P, R, F1 = bert_scorer.score([candidate_summary], [reference_summary])
        bert_f1_score = F1.mean().item()

        # Calculate AlignScore
        align_score = align_scorer.score(contexts=[opinion_of_the_court], claims=[candidate_summary])[0]

        # Prepare data for UniEval
        data = convert_to_json(output_list=[candidate_summary], 
                               src_list=[opinion_of_the_court], 
                               ref_list=[reference_summary])

        # Get UniEval scores
        eval_scores = evaluator.evaluate(data, print_result=False)

        # Extract individual scores from eval_scores
        coherence_score = eval_scores[0]['coherence']
        consistency_score = eval_scores[0]['consistency']
        fluency_score = eval_scores[0]['fluency']
        relevance_score = eval_scores[0]['relevance']
        overall_score = eval_scores[0]['overall']

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
            'bert_f1': bert_f1_score,
            'align_score': align_score,
            'unieval_coherence': coherence_score,
            'unieval_consistency': consistency_score,
            'unieval_fluency': fluency_score,
            'unieval_relevance': relevance_score,
            'unieval_overall': overall_score
        })

        logging.debug(f"Processed row {index}")

    # Convert the scores list to a DataFrame
    scores_df = pd.DataFrame(scores_list)

    # Save the DataFrame to a CSV file
    scores_df.to_csv(csv_filename, index=False)
    logging.info(f"Scores saved to {csv_filename}")
    print(f"Scores saved to {csv_filename}")
