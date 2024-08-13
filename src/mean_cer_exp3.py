import json
import os

from src.utils.constants import automated_resuts


def calculate_mean_cer(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    ocr_cer_values = []
    check_label_cer_values = []

    for entry in data:
        if 'OCR' in entry:
            ocr_cer_values.append(entry['OCR']['cer'])
        if 'Check Label from Mistral' in entry:
            check_label_cer_values.append(entry['Check Label from Mistral']['cer'])

    mean_ocr_cer = round(sum(ocr_cer_values) / len(ocr_cer_values), 3) if ocr_cer_values else None
    mean_check_label_cer = round(sum(check_label_cer_values) / len(check_label_cer_values),
                                 3) if check_label_cer_values else None

    return mean_ocr_cer, mean_check_label_cer


# Example usage:
json_file_path = os.path.join(automated_resuts, 'final_exp_3.json')
mean_ocr_cer, mean_check_label_cer = calculate_mean_cer(json_file_path)

if mean_ocr_cer is not None:
    print(f"Mean OCR CER value: {mean_ocr_cer}")
else:
    print("No OCR CER values found")

if mean_check_label_cer is not None:
    print(f"Mean Check Label from Mistral CER value: {mean_check_label_cer}")
else:
    print("No Check Label from Mistral CER values found")
