import json
import os

from src.utils.constants import automated_resuts_experiments


def calculate_label_correctness_and_mean_cer(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    check_label_1_correct_count = 0
    check_label_1_incorrect_count = 0
    check_label_2_correct_count = 0
    check_label_2_incorrect_count = 0
    ocr_cer_values = []
    check_label_1_cer_values = []
    check_label_2_cer_values = []

    for entry in data:
        # Calculate correctness counts for Check Label-1
        if entry['Check Label-1']['is_correct_1'] == "Yes":
            check_label_1_correct_count += 1
        else:
            check_label_1_incorrect_count += 1

        # Calculate correctness counts for Check Label-2
        if entry['Check Label-2']['is_correct_2'] == "Yes":
            check_label_2_correct_count += 1
        else:
            check_label_2_incorrect_count += 1

        # Collect CER values
        ocr_cer_values.append(entry['OCR']['cer'])
        check_label_1_cer_values.append(entry['Check Label-1']['cer'])
        check_label_2_cer_values.append(entry['Check Label-2']['cer'])

    mean_ocr_cer = round((sum(ocr_cer_values) / len(ocr_cer_values))*100, 3)
    mean_check_label_1_cer = round((sum(check_label_1_cer_values) / len(check_label_1_cer_values))*100, 3)
    mean_check_label_2_cer = round((sum(check_label_2_cer_values) / len(check_label_2_cer_values))*100, 3)

    return {
        "check_label_1_correct_count": check_label_1_correct_count,
        "check_label_1_incorrect_count": check_label_1_incorrect_count,
        "check_label_2_correct_count": check_label_2_correct_count,
        "check_label_2_incorrect_count": check_label_2_incorrect_count,
        "mean_ocr_cer": mean_ocr_cer,
        "mean_check_label_1_cer": mean_check_label_1_cer,
        "mean_check_label_2_cer": mean_check_label_2_cer
    }


# Example usage:
# Replace 'your_json_file_path.json' with the path to your actual JSON file
json_file_path = os.path.join(automated_resuts_experiments, 'final_exp_6_v5.json')
results = calculate_label_correctness_and_mean_cer(json_file_path)
print(f"First check : Correct labels: {results['check_label_1_correct_count']} | Incorrect labels: {results['check_label_1_incorrect_count']}")
print(f"Second check: Correct labels: {results['check_label_2_correct_count']} | Incorrect labels: {results['check_label_2_incorrect_count']}")
print(f"Mean OCR CER: {results['mean_ocr_cer']}")
print(f"Mean CER - First check: {results['mean_check_label_1_cer']}")
print(f"Mean CER - Second check: {results['mean_check_label_2_cer']}")
