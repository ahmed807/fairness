import sled  # ** required so SLED would be properly registered by the AutoClasses **
from transformers import AutoTokenizer, AutoModel

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('tau/bart-base-sled')
model = AutoModel.from_pretrained('tau/bart-base-sled')

# Input text
text = "Hello, my dog is cute"

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Get model outputs
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

# Print the original input
print("Original text:", text)

# Print the token IDs
print("\nToken IDs:", inputs['input_ids'][0].tolist())

# Decode the entire sequence
decoded_text = tokenizer.decode(inputs['input_ids'][0])
print("\nDecoded full text:", decoded_text)

# Print individual tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("\nIndividual tokens:", tokens)

# Print the last hidden states shape
print("\nLast hidden states shape:", last_hidden_states.shape)
print("\nLast hidden states:", last_hidden_states)
