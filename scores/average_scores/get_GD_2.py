import os
import pandas as pd
import numpy as np

def get_all_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files

def process_issue_area(file_path):
    df = pd.read_csv(file_path)
    if 'issueArea' in df.columns:
        df = df[(df['issueArea'] >= 1) & (df['issueArea'] <= 5)]
    return df

def process_vote_distribution(file_path):
    df = pd.read_csv(file_path)
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['50% - 60%', '60% - 70%', '70% - 80%', '80% - 90%', '90% - 100%']
    df['voteRange'] = pd.cut(df['voteDistribution'], bins=bins, labels=labels, right=True)
    averaged_df = df.groupby('voteRange').mean().reset_index()
    averaged_df.drop(columns=['voteDistribution'], inplace=True)
    return averaged_df

def calculate_gd_for_metrics(df):
    # Ignore the first column for GD calculation
    df = df.iloc[:, 1:]
    
    gd_values = {}
    for metric in df.columns:
        values = df[metric].values * 100
        values = np.round(values, 2)
        
        average = np.mean(values)
        squared_differences = (values - average) ** 2
        sum_squared_differences = np.sum(squared_differences)
        gd = np.sqrt(sum_squared_differences / len(values))
        gd_values[metric] = gd
    
    return gd_values

def extract_folder_name(file_path):
    # Extract the folder name part after 'average_scores_case_scores_'
    folder_name = os.path.basename(file_path)
    start = folder_name.find('average_scores_case_scores_') + len('average_scores_case_scores_')
    end = folder_name.find('.csv')
    return folder_name[start:end]

def process_csv_files(directory):
    results = []
    csv_files = get_all_csv_files(directory)
    x = 0

    for file_path in csv_files:
        if 'issueArea' in file_path:
            df = process_issue_area(file_path)
        elif 'voteDistribution' in file_path:
            df = process_vote_distribution(file_path)
        else:
            df = pd.read_csv(file_path)
        
        if x <= 7:
            print(file_path)
            x+=1
        gd_values = calculate_gd_for_metrics(df)
        folder_name = extract_folder_name(file_path)
        
        result_row = [folder_name] + [gd_values[metric] for metric in gd_values]
        results.append(result_row)
    
    header = ['Folder Name'] + list(gd_values.keys())
    
    output_df = pd.DataFrame(results, columns=header)
    output_df.to_csv('/home/mostah/workspace/fairness/scores/GD_results/new/group_disparity_results.csv', index=False)

directory_path = '/home/mostah/workspace/fairness/scores/average_scores'
process_csv_files(directory_path)
