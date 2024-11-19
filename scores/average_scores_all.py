import pandas as pd
import json
import os

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_output_path(base_dir, csv_filename, key):
    """Generate output path maintaining the original folder structure"""
    # Extract the folder name from csv_filename (e.g., 'top_k', 'temperature', etc.)
    folder_name = os.path.dirname(csv_filename).split('/')[0]
    
    # Create the output directory structure
    output_dir = os.path.join(base_dir, folder_name)
    create_directory(output_dir)
    
    # Generate the output filename
    filename = os.path.basename(csv_filename)
    output_filename = f'average_scores_{filename}_{key}.csv'
    return os.path.join(output_dir, output_filename)

def calculate_average_scores(df, keys, score_columns):
    average_scores = {}
    for key in keys:
        group_means = df.groupby(key).mean()
        average_scores[key] = group_means[score_columns]
    return average_scores

def process_files(config_file):
    # Create main output directory
    output_base_dir = "average_scores"
    create_directory(output_base_dir)
    
    # List of keys and score columns
    keys = ['issueArea', 'decisionDirection', 'partyWinning', 'voteDistribution', 'respondentType']
    score_columns = [
        'rouge1', 'rouge2', 'rougeL', 'bert_f1',
        'align_score', 'unieval_coherence', 'unieval_consistency',
        'unieval_fluency', 'unieval_relevance', 'unieval_overall'
    ]
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Dictionary to store results for each file
    all_results = {}
    
    # Process each file in the config
    for entry in config:
        csv_filename = entry['csv_filename']
        try:
            # Read the CSV file
            df = pd.read_csv(csv_filename)
            
            # Calculate average scores for this file
            file_results = calculate_average_scores(df, keys, score_columns)
            
            # Store results with the filename as key
            all_results[csv_filename] = file_results
            
            # Print results for this file
            print(f"\nResults for {csv_filename}:")
            for key, scores in file_results.items():
                print(f"\nAverage scores for {key}:")
                print(scores)
                
                # Save individual results to CSV in appropriate subdirectory
                output_path = get_output_path(output_base_dir, csv_filename, key)
                scores.to_csv(output_path)
                print(f"Saved results to: {output_path}")
                
        except FileNotFoundError:
            print(f"Warning: File {csv_filename} not found")
        except Exception as e:
            print(f"Error processing {csv_filename}: {str(e)}")
    
    return all_results

# Execute the analysis
if __name__ == "__main__":
    results = process_files('all_scores_config copy.json')
