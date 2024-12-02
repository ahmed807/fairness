import os

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

def fix_filenames(directory):
    # Get all CSV files
    csv_files = get_all_csv_files(directory)
    
    for file_path in csv_files:
        # Get the directory and filename
        dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        # Remove the unwanted '.csv' in the middle of the filename
        new_file_name = file_name.replace('.csv_', '_')
        
        # Construct the new file path
        new_file_path = os.path.join(dir_name, new_file_name)
        
        # Rename the file
        if new_file_path != file_path:  # Check if the name actually changes
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} -> {new_file_path}')

# Specify the directory containing the CSV files
directory_path = '/home/mostah/workspace/fairness/scores/average_scores'
fix_filenames(directory_path)
