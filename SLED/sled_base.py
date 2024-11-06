import sled  # ** required so SLED would be properly registered by the AutoClasses **
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('tau/bart-base-sled')
model = AutoModel.from_pretrained('tau/bart-base-sled')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

