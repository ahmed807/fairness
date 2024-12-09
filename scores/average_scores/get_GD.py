import os
import pandas as pd
import numpy as np

def get_all_csv_files(directory):
    csv_files = []
    # Traverse the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a CSV
            if file.endswith('.csv'):
                # Construct the full file path
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files

def calculate_gd_for_metrics(file_path):
    # Read the CSV file, ignoring the first column
    df = pd.read_csv(file_path).iloc[:, 1:]
    
    # Dictionary to store GD for each metric
    gd_values = {}
    
    # Calculate GD for each metric
    for metric in df.columns:
        # Convert values to percentage and round to two decimals
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
    x= 1
    
    # Get all CSV files
    csv_files = get_all_csv_files(directory)
    
    for file_path in csv_files:
        if x == 1:
            print(file_path)
            x+=1
        gd_values = calculate_gd_for_metrics(file_path)
        folder_name = extract_folder_name(file_path)
        
        # Prepare the result row
        result_row = [folder_name] + [gd_values[metric] for metric in gd_values]
        results.append(result_row)
    
    # Define the header for the output CSV
    header = ['Folder Name'] + list(gd_values.keys())
    
    # Write the results to a new CSV file
    output_df = pd.DataFrame(results, columns=header)
    output_df.to_csv('/home/mostah/workspace/fairness/scores/GD_results/group_disparity_results.csv', index=False)

# Specify the directory containing the CSV files
directory_path = '/home/mostah/workspace/fairness/scores/average_scores'
process_csv_files(directory_path)
