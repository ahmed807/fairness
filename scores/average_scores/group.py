import os
import pandas as pd

def get_all_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files

def extract_last_name(file_path):
    file_name = os.path.basename(file_path)
    last_name = file_name.split('_')[-1].replace('.csv', '')
    return last_name

def extract_folder_name(file_path):
    return os.path.basename(os.path.dirname(file_path))

def process_csv_files(directory):
    grouped_files = {}
    
    csv_files = get_all_csv_files(directory)
    
    for file_path in csv_files:
        last_name = extract_last_name(file_path)
        folder_name = extract_folder_name(file_path)
        
        if last_name not in grouped_files:
            grouped_files[last_name] = []
        
        grouped_files[last_name].append((file_path, folder_name))
    
    for last_name, files in grouped_files.items():
        dataframes = []
        for file_path, folder_name in files:
            df = pd.read_csv(file_path)
            df.insert(0, 'Folder Name', folder_name)  # Insert the folder name as the first column
            dataframes.append(df)
        
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        
        output_path = f"/home/mostah/workspace/fairness/scores/average_scores/grouped/{last_name}.csv"
        concatenated_df.to_csv(output_path, index=False)
        print(f"Saved concatenated file to {output_path}")

# Specify the directory containing the CSV files
directory_path = '/home/mostah/workspace/fairness/scores/average_scores'
process_csv_files(directory_path)
