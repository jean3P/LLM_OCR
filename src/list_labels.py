import json
import os

from src.utils.constants import results_LLM_mistral_2

save_mistral_output = os.path.join(results_LLM_mistral_2, 'evaluation_results_with_mistral_1.json')
# Read data from file:
with open(save_mistral_output) as f:
    data = json.load(f)

ground_truth_labels = []
ocr_predicted_labels = []

for item in data:
    # Append labels to respective lists
    ground_truth_labels.append(item['ground_truth_label'])
    ocr_predicted_labels.append(item['OCR']['predicted_label'])

print("Ground Truth Labels:")
print("\n".join(ground_truth_labels))
print('-' * 100)
print("\n\nOCR Predicted Labels:")
print("\n".join(ocr_predicted_labels))