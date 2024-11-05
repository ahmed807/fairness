import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('case_scores_mixed.csv')

# List of keys to calculate average scores for
keys = ['issueArea', 'decisionDirection', 'partyWinning', 'voteDistribution', 'respondentType']

# List of score columns to calculate averages for
score_columns = [
    'rouge1', 'rouge2', 'rougeL', 'bert_f1',
    'align_score', 'unieval_coherence', 'unieval_consistency',
    'unieval_fluency', 'unieval_relevance', 'unieval_overall'
]

# Dictionary to store average scores for each key
average_scores = {}

# Calculate average scores for each key
for key in keys:
    # Group by the key and calculate the mean for each group
    group_means = df.groupby(key).mean()

    # Store the results in the dictionary
    average_scores[key] = group_means[score_columns]

    # Print the results
    print(f"Average scores for {key}:")
    print(average_scores[key])
    print("\n")

# Optionally, save the average scores to a new CSV file
for key, scores in average_scores.items():
    scores.to_csv(f'./scores/mixed/average_scores_mixed_{key}.csv')
