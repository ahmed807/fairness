import os
import glob
import shutil

# Define the path to the directory containing the CSV files
source_directory = '/home/mostah/workspace/fairness/scores/average_scores/temperature'

# Use glob to find all CSV files in the directory
csv_files = glob.glob(os.path.join(source_directory, '*.csv'))

# Iterate over each file
for file_path in csv_files:
    # Extract the base name of the file
    file_name = os.path.basename(file_path)
    
    # Extract the folder name from the file name
    # Assuming the pattern is case_scores_<name>.csv_
    if 'case_scores_' in file_name and '.csv_' in file_name:
        start = file_name.find('case_scores_') + len('case_scores_')
        end = file_name.find('.csv_')
        folder_name = file_name[start:end]
        
        # Define the new directory path
        new_directory = os.path.join(source_directory, folder_name)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        
        # Move the file to the new directory
        shutil.move(file_path, os.path.join(new_directory, file_name))

print("Files have been organized into folders.")