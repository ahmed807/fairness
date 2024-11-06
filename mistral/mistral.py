# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3").to('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Set pad_token_id to eos_token_id if not already set
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# # Function to predict decision direction
# def predict_decision_direction(summary):
#     # Define the prompt for classification
#     prompt = f'''In order to determine whether an outcome is liberal (=2) or conservative (=1), the following scheme is employed. 1. In the context of issues pertaining to criminal procedure, civil rights, First Amendment, due process, privacy, and attorneys, liberal (2)= pro-person accused or convicted of crime, or denied a jury trial pro-civil liberties or civil rights claimant, especially those exercising less protected civil rights (e.g., homosexuality) pro-child or juvenile pro-indigent pro-Indian pro-affirmative action pro-neutrality in establishment clause cases pro-female in abortion pro-underdog anti-slavery incorporation of foreign territories anti-government in the context of due process, except for takings clause cases where a pro-government, anti-owner vote is considered liberal except in criminal forfeiture cases or those where the taking is pro-business violation of due process by exercising jurisdiction over nonresident pro-attorney or governmental official in non-liability cases pro-accountability and/or anti-corruption in campaign spending pro-privacy vis-a-vis the 1st Amendment where the privacy invaded is that of mental incompetents pro-disclosure in Freedom of Information Act issues except for employment and student records conservative (1)=the reverse of above 2. In the context of issues pertaining to unions and economic activity, liberal (2)= pro-union except in union antitrust where liberal = pro-competition pro-government anti-business anti-employer pro-competition pro-injured person pro-indigent pro-small business vis-a-vis large business pro-state/anti-business in state tax cases pro-debtor pro-bankrupt pro-Indian pro-environmental protection pro-economic underdog pro-consumer pro-accountability in governmental corruption pro-original grantee, purchaser, or occupant in state and territorial land claims anti-union member or employee vis-a-vis union anti-union in union antitrust anti-union in union or closed shop pro-trial in arbitration conservative (1)= reverse of above 3. In the context of issues pertaining to judicial power, liberal (2)= pro-exercise of judicial power pro-judicial "activism" pro-judicial review of administrative action conservative (1)=reverse of above 4. In the context of issues pertaining to federalism, liberal (2)= pro-federal power pro-executive power in executive/congressional disputes anti-state conservative (1)=reverse of above 5. In the context of issues pertaining to federal taxation, liberal (2)= pro-United States; conservative (1)= pro-taxpayer 6. In interstate relations and private law issues, unspecifiable (3) for all such cases. 7. In miscellaneous, incorporation of foreign territories and executive authority vis-a-vis congress or the states or judicial authority vis-a-vis state or federal legislative authority = (2); legislative veto = (1). Values: 1 conservative 2 liberal 3 unspecifiable Classify the following summary as liberal or conservative:

#  Classify the following summary as liberal or conservative:\n\n{summary}\n\nDecision Direction:'''

#     # Tokenize the input with padding and truncation
#     inputs = tokenizer(
#         prompt,
#         return_tensors='pt',
#         padding=True,
#         truncation=True,
#         max_length=4096  # Set a reasonable max_length
#     ).to(model.device)

#     # Generate prediction with attention mask
#     outputs = model.generate(
#         inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_new_tokens=50,
#         num_return_sequences=1,
#         pad_token_id=tokenizer.pad_token_id
#     )

#     # Decode the output
#     prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Extract the decision direction from the prediction
#     if "liberal" in prediction.lower():
#         return "liberal"
#     elif "conservative" in prediction.lower():
#         return "conservative"
#     else:
#         return "unknown"

# # Example usage
# generated_summary = '''To address the large number of aliens within its borders who do not have a lawful right to be in this country, Arizona enacted a statute (S. B. 1070) that makes it a misdemeanor for an unauthorized alien to seek or engage in work in the State, and that provides that officers who conduct a stop, detention, or arrest must in some circumstances make efforts to verify the person's immi- gration status with the Federal Government. The statute establishes an official state policy of attrition through enforcement, and provides that, once an alien is removed from the United States, he or she has probable cause to believe that he has committed any public offense that makes the person removable. Section 5(C) of the statute provides that a state officer, without a warrant, may arrest a person if the officer has reasonable suspicion that his presence in the country is unlawful. Other provisions give specific arrest authority and inves- tigative duties with respect to certain aliens to state and local law enforcement officers. The United States filed suit in Federal District Court to enjoin the four provisions at issue from taking effect, and the Court of Appeals affirmed. Held: The federal law preempts and renders invalid four separate provisions of the state law. . (a) Under pre- emption principles, federal law permits Arizona to implement the state-law provisions in dispute. P.. (b) Congress has the power to preempt state law by enacting a statute containing an express preemption provision. Congress may withdraw specified powers from the States in enacting such a statute, and state law must also give way to federal law in at least two other circumstances. First, the States are precluded from regulating conduct in a field that Congress, acting within its proper authority, has determined must be regulated by its exclusive governance. Second, state laws are preempted when they conflict with federal law, which includes cases where compliance with both federal and state regulations is a physical impossibility. Here, the unilateral state action to detain authorized by §6 goes far beyond these measures, defeating any need for real cooperation. Moreover, § 6 attempts to provide state officers even greater authority to arrest aliens on the basis of possible removability than Congress has given to trained federal immi gration officers. Thus, §6 creates an obstacle to the enforcement of federal immigration law, and is preempted by federal law.. 641 F.3d 339, affirmed in part and reversed in part. Justice Kagan took no part in the consideration or decision of this case.'''
# decision_direction = predict_decision_direction(generated_summary)
# print(f"The predicted decision direction is: {decision_direction}")






from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from huggingface_hub import login, HfApi
login(token='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')
logging.basicConfig(filename='mistral_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
dataset = load_dataset("ahmed275/opinions_dataset_temporal_test_generated_summaries")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3").to("cuda")

# Set pad_token_id to eos_token_id if not already set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id



# Function to predict decision direction
def predict_decision_direction(summary):
    prompt = f'''In order to determine whether an outcome is liberal (2) or conservative (1), the following scheme is employed. 
    1. In the context of issues pertaining to criminal procedure, civil rights, First Amendment, due process, privacy, and attorneys, liberal (2)= pro-person accused or convicted of crime, or denied a jury trial pro-civil liberties or civil rights claimant, especially those exercising less protected civil rights (e.g., homosexuality) pro-child or juvenile pro-indigent pro-Indian pro-affirmative action pro-neutrality in establishment clause cases pro-female in abortion pro-underdog anti-slavery incorporation of foreign territories anti-government in the context of due process, except for takings clause cases where a pro-government, anti-owner vote is considered liberal except in criminal forfeiture cases or those where the taking is pro-business violation of due process by exercising jurisdiction over nonresident pro-attorney or governmental official in non-liability cases pro-accountability and/or anti-corruption in campaign spending pro-privacy vis-a-vis the 1st Amendment where the privacy invaded is that of mental incompetents pro-disclosure in Freedom of Information Act issues except for employment and student records conservative (1)=the reverse of above 
    2. In the context of issues pertaining to unions and economic activity, liberal (2)= pro-union except in union antitrust where liberal = pro-competition pro-government anti-business anti-employer pro-competition pro-injured person pro-indigent pro-small business vis-a-vis large business pro-state/anti-business in state tax cases pro-debtor pro-bankrupt pro-Indian pro-environmental protection pro-economic underdog pro-consumer pro-accountability in governmental corruption pro-original grantee, purchaser, or occupant in state and territorial land claims anti-union member or employee vis-a-vis union anti-union in union antitrust anti-union in union or closed shop pro-trial in arbitration conservative (1)= reverse of above 
    3. In the context of issues pertaining to judicial power, liberal (2)= pro-exercise of judicial power pro-judicial "activism" pro-judicial review of administrative action conservative (1)=reverse of above 
    4. In the context of issues pertaining to federalism, liberal (2)= pro-federal power pro-executive power in executive/congressional disputes anti-state conservative (1)=reverse of above 
    5. In the context of issues pertaining to federal taxation, liberal (2)= pro-United States; conservative (1)= pro-taxpayer 
    6. In interstate relations and private law issues, unspecifiable (3) for all such cases. 
    7. In miscellaneous, incorporation of foreign territories and executive authority vis-a-vis congress or the states or judicial authority vis-a-vis state or federal legislative authority = (2); legislative veto = (1). 
    Values: 1 conservative, 2 liberal, 3 unspecifiable.
    Classify the following summary as liberal or conservative: \n\n{summary}\n\n Decision Direction: '''
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Determine the prediction
    if "liberal" in prediction.lower():
        predicted_label = 2.0
    elif "conservative" in prediction.lower():
        predicted_label = 1.0
    else:
        predicted_label = 3.0
    
    # Log the summary and prediction if the counter is less than 10
    if log_counter < 3:
        logging.info("Summary: %s", summary)
        logging.info("Predicted Decision Direction: %f", predicted_label)
        log_counter += 1
    
    return predicted_label



# Extract summaries and true labels
summaries_opinion = dataset['train']['opinionOfTheCourt']
summaries_syllabus = dataset['train']['syllabus']
true_labels = dataset['train']['decisionDirection']

# Predict decision directions for opinionOfTheCourt
log_counter = 0
predicted_labels_opinion = [predict_decision_direction(summary) for summary in summaries_opinion]

# Calculate metrics for opinionOfTheCourt
accuracy_opinion = accuracy_score(true_labels, predicted_labels_opinion)
precision_opinion = precision_score(true_labels, predicted_labels_opinion, average='macro', zero_division=0)
recall_opinion = recall_score(true_labels, predicted_labels_opinion, average='macro', zero_division=0)
f1_opinion = f1_score(true_labels, predicted_labels_opinion, average='macro', zero_division=0)

print(f"Opinion - Accuracy: {accuracy_opinion}")
print(f"Opinion - Precision: {precision_opinion}")
print(f"Opinion - Recall: {recall_opinion}")
print(f"Opinion - F1 Score: {f1_opinion}")
logging.info("Opinion - Accuracy: %f", accuracy_opinion)
logging.info("Opinion - Precision: %f", precision_opinion)
logging.info("Opinion - Recall: %f", recall_opinion)
logging.info("Opinion - F1 Score: %f", f1_opinion)

# Predict decision directions for syllabus
log_counter = 0
predicted_labels_syllabus = [predict_decision_direction(summary) for summary in summaries_syllabus]

# Calculate metrics for syllabus
accuracy_syllabus = accuracy_score(true_labels, predicted_labels_syllabus)
precision_syllabus = precision_score(true_labels, predicted_labels_syllabus, average='macro', zero_division=0)
recall_syllabus = recall_score(true_labels, predicted_labels_syllabus, average='macro', zero_division=0)
f1_syllabus = f1_score(true_labels, predicted_labels_syllabus, average='macro', zero_division=0)

print(f"Syllabus - Accuracy: {accuracy_syllabus}")
print(f"Syllabus - Precision: {precision_syllabus}")
print(f"Syllabus - Recall: {recall_syllabus}")
print(f"Syllabus - F1 Score: {f1_syllabus}")
logging.info("Syllabus - Accuracy: %f", accuracy_syllabus)
logging.info("Syllabus - Precision: %f", precision_syllabus)
logging.info("Syllabus - Recall: %f", recall_syllabus)
logging.info("Syllabus - F1 Score: %f", f1_syllabus)
