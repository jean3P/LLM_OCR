import json
import os

from src.utils.constants import automated_resuts


def calculate_mean_cer_exp1(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    ocr_cer_values = []
    mistral_cer_values = []
    select_cer_values = []

    for entry in data:
        ocr_cer_values.append(entry['OCR']['cer'])
        mistral_cer_values.append(entry['MISTRAL']['cer'])
        select_cer_values.append(entry['Select the best']['cer'])

    mean_ocr_cer = round(sum(ocr_cer_values) / len(ocr_cer_values), 3)
    mean_mistral_cer = round(sum(mistral_cer_values) / len(mistral_cer_values), 3)
    mean_select_cer = round(sum(select_cer_values) / len(select_cer_values), 3)

    return mean_ocr_cer, mean_mistral_cer, mean_select_cer

# Example usage:
json_file_path = os.path.join(automated_resuts, 'final_exp_1.json')
mean_ocr_cer, mean_mistral_cer, mean_select_cer = calculate_mean_cer_exp1(json_file_path)
print(f"Mean OCR CER value: {mean_ocr_cer}")
print(f"Mean MISTRAL CER value: {mean_mistral_cer}")
print(f"Mean Select the Best CER value: {mean_select_cer}")
