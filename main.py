



import subprocess
import logging
import yaml
import os
# Configure logging
logging.basicConfig(
    filename='main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define model and tokenizer configurations
model_tokenizer_configs = [
    {
        "model": "LongT5ForConditionalGeneration.from_pretrained('Stancld/longt5-tglobal-large-16384-pubmed-3k_steps')",
        "tokenizer": "AutoTokenizer.from_pretrained('Stancld/longt5-tglobal-large-16384-pubmed-3k_steps')",
        "log_file_name" : "t5-large_log.log",
        "pretrained_model_save_path" : "LongT5_tglobal_large/model",
        "repo_id": "ahmed275/LongT5_tglobal_large",
        "push_dataset_name": "ahmed275/generated_summaries_LongT5_tglobal_large"


    },
    {
        "tokenizer" : "AutoTokenizer.from_pretrained('Stancld/long-t5-tglobal-base')",
        "model" : "AutoModelForSeq2SeqLM.from_pretrained('Stancld/long-t5-tglobal-base')",
        "log_file_name" : "t5-base_log.log",
        "pretrained_model_save_path" : "LongT5_tglobal_base/model",
        "repo_id": "ahmed275/LongT5_tglobal_base",
        "push_dataset_name": "ahmed275/generated_summaries_LongT5_tglobal_base"

                
    }
]

def update_config(config_path, model, tokenizer,log_file_name,pretrained_model_save_path,repo_id,push_dataset_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the model and tokenizer in the config
    config['led']['model'] = model
    config['led']['tokenizer'] = tokenizer
    config['led']['log_file_name'] = log_file_name
    config['led']['pretrained_model_save_path'] = pretrained_model_save_path
    config['led']['repo_id'] = repo_id
    config['generate_summary']['model'] = model
    config['generate_summary']['tokenizer'] = tokenizer
    config['generate_summary']['push_dataset_name'] = push_dataset_name

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

def run_script(script_name, config_path, cwd=None):
    try:
        # Run the script using subprocess with specified working directory and config path
        result = subprocess.run(['python3', script_name, config_path], check=True, cwd=cwd)
        logging.info(f"{script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running {script_name}: {e}")
        print(f"An error occurred while running {script_name}: {e}")

def main():
    config_path = os.path.abspath('./config.yaml')

    for config in model_tokenizer_configs:
        # Update the configuration file with the current model and tokenizer and log_file_name
        update_config(config_path, config['model'], config['tokenizer'], config['log_file_name'],config['pretrained_model_save_path'], config['repo_id'], config['push_dataset_name'])

        # Run LED.py located in the LED folder
        logging.info(f"Running LED.py with model {config['model']}...")
        print(f"Running LED.py with model {config['model']}...")
        run_script('LED.py', config_path, cwd='LED')


        logging.info("Running generate_summary.py with initial configuration...")
        print("Running generate_summary.py with initial configuration...")
        run_script('generate_summaries.py', config_path, cwd='generate_summaries')

if __name__ == "__main__":
    main()
