



import subprocess
import logging
import yaml

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
        "tokenizer": "AutoTokenizer.from_pretrained('Stancld/longt5-tglobal-large-16384-pubmed-3k_steps')"
    },
    # {
    #     "model": "LongT5ForConditionalGeneration.from_pretrained('Stancld/LongT5-TGlobal-Base')",
    #     "tokenizer": "AutoTokenizer.from_pretrained('Stancld/LongT5-TGlobal-Base')"
    # },
    # {
    #     "model": "LongT5ForConditionalGeneration.from_pretrained('Stancld/LongT5-TGlobal-XL')",
    #     "tokenizer": "AutoTokenizer.from_pretrained('Stancld/LongT5-TGlobal-XL')"
    # }
]

def update_config(config_path, model, tokenizer):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the model and tokenizer in the config
    config['led']['model'] = model
    config['led']['tokenizer'] = tokenizer

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
    config_path = 'config.yaml'

    for config in model_tokenizer_configs:
        # Update the configuration file with the current model and tokenizer
        update_config(config_path, config['model'], config['tokenizer'])

        # Run LED.py located in the LED folder
        logging.info(f"Running LED.py with model {config['model']}...")
        print(f"Running LED.py with model {config['model']}...")
        run_script('LED.py', config_path, cwd='LED')


        logging.info("Running generate_summary.py with initial configuration...")
        print("Running generate_summary.py with initial configuration...")
        run_script('generate_summaries.py', config_path, cwd='generate_summaries')

if __name__ == "__main__":
    main()
