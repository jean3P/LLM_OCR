import json
import os

from src.utils.constants import results_mixed_LLM_MISTRAL, automated_resuts

# Specify the directory and file name

file_name = 'final_test_evaluation_from_mistral_50_50.json'
save_mistral_output = os.path.join(automated_resuts, file_name)

# Load the JSON data from file
with open(save_mistral_output, 'r') as file:
    json_data = json.load(file)

# Initialize counters for the analysis
same_label_cer_0_count = 0
modified_label_cer_0_count = 0
same_label_cer_greater_than_0_count = 0
modified_label_cer_greater_than_0_count = 0

# Process each item in the JSON data
for item in json_data:
    ocr_cer = item["OCR"]["cer"]
    ocr_label = item["OCR"]["predicted_label"]
    mistral_label = item["MISTRAL"]["predicted_label"]

    # Determine if the label remained the same or was modified by Mistral
    label_changed = (ocr_label != mistral_label)

    # Count based on CER value and whether the label was changed
    if ocr_cer == 0:
        if label_changed:
            modified_label_cer_0_count += 1
        else:
            same_label_cer_0_count += 1
    else:
        if label_changed:
            modified_label_cer_greater_than_0_count += 1
        else:
            same_label_cer_greater_than_0_count += 1

# Calculate total counts for normalization
total_cer_0 = same_label_cer_0_count + modified_label_cer_0_count
total_cer_greater_than_0 = same_label_cer_greater_than_0_count + modified_label_cer_greater_than_0_count

# Calculate percentages
same_label_cer_0_percentage = (same_label_cer_0_count / total_cer_0 * 100) if total_cer_0 else 0
modified_label_cer_0_percentage = (modified_label_cer_0_count / total_cer_0 * 100) if total_cer_0 else 0
same_label_cer_greater_than_0_percentage = (same_label_cer_greater_than_0_count / total_cer_greater_than_0 * 100) if total_cer_greater_than_0 else 0
modified_label_cer_greater_than_0_percentage = (modified_label_cer_greater_than_0_count / total_cer_greater_than_0 * 100) if total_cer_greater_than_0 else 0

# Output the table
print("OCR Mean CER\t| Same label in Mistral\t| Modified in Mistral")
print("-------------------------------------------------------------")
print(f"CER = 0.0\t\t| {same_label_cer_0_percentage:.2f}%\t\t\t\t| {modified_label_cer_0_percentage:.2f}%")
print(f"CER > 0  \t\t| {same_label_cer_greater_than_0_percentage:.2f}%\t\t\t\t| {modified_label_cer_greater_than_0_percentage:.2f}%")