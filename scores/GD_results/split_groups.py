import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('group_disparity_results.csv')

# Define the categories to split by
categories = ['partyWinning', 'respondentType', 'issueArea', 'voteDistribution', 'decisionDirection']

# Iterate over each category and create a separate CSV file
for category in categories:
    # Filter the DataFrame for rows containing the category in the 'Folder Name' column
    filtered_df = df[df['Folder Name'].str.contains(category)]
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(f'{category}.csv', index=False)

print("CSV files have been successfully created for each category.")
