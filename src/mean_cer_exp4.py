import json
import os

from src.utils.constants import automated_resuts


def calculate_mean_cer(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    ocr_cer_values = []
    select_best_mistral_cer_values = []

    for entry in data:
        if 'OCR' in entry:
            ocr_cer_values.append(entry['OCR']['cer'])
            # print(f"OCR CER value added: {entry['OCR']['cer']}")
        if 'Select best Mistral label' in entry:
            select_best_mistral_cer_values.append(entry['Select best Mistral label']['cer'])
            # print(f"Select best Mistral label CER value added: {entry['Select best Mistral label']['cer']}")

    mean_ocr_cer = round(sum(ocr_cer_values) / len(ocr_cer_values), 3) if ocr_cer_values else None
    mean_select_best_mistral_cer = round(sum(select_best_mistral_cer_values) / len(select_best_mistral_cer_values), 3) if select_best_mistral_cer_values else None

    return mean_ocr_cer, mean_select_best_mistral_cer


# Example usage:
json_file_path = os.path.join(automated_resuts, 'final_exp_5.json')
mean_ocr_cer, mean_select_best_mistral_cer = calculate_mean_cer(json_file_path)

if mean_ocr_cer is not None:
    print(f"Mean OCR CER value: {mean_ocr_cer}")
else:
    print("No OCR CER values found")

if mean_select_best_mistral_cer is not None:
    print(f"Mean Select best Mistral label CER value: {mean_select_best_mistral_cer}")
else:
    print("No Select best Mistral label CER values found")
