from bert_score import score

candidates = ["The dog quickly ran across the park."]
references = ["A quick brown dog raced through the park."]

precision, recall, f1 = score(candidates, references, lang='en',device='cuda:0', batch_size=16)

print(f'Precision: {precision.mean():.4f}')
print(f'Recall: {recall.mean():.4f}')
print(f'F1 Score: {f1.mean():.4f}')
