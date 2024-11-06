
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('./all/average_scores_voteDistribution.csv')

# Define the bins and labels for the vote distribution ranges
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['50% - 60%', '60% - 70%', '70% - 80%', '80% - 90%', '90% - 100%']

# Categorize the voteDistribution into the specified ranges
df['voteRange'] = pd.cut(df['voteDistribution'], bins=bins, labels=labels, right=True)
print(df)
# Group by the voteRange and calculate the mean for each group
averaged_df = df.groupby('voteRange').mean().reset_index()

# Drop the original voteDistribution column
averaged_df.drop(columns=['voteDistribution'], inplace=True)

# Save the result to a new CSV file
averaged_df.to_csv('averaged_output.csv', index=False)

print("Averaged data saved to 'averaged_output.csv'.")
