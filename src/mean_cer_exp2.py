import json
import os

from src.utils.constants import automated_resuts


def calculate_mean_cer(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    ocr_cer_values = []
    check_label_cer_values = []

    for entry in data:
        ocr_cer_values.append(entry['OCR']['cer'])
        check_label_cer_values.append(entry['Check Label']['cer'])

    mean_ocr_cer = round(sum(ocr_cer_values) / len(ocr_cer_values), 3)
    mean_check_label_cer = round(sum(check_label_cer_values) / len(check_label_cer_values), 3)

    return mean_ocr_cer, mean_check_label_cer


# Example usage:
json_file_path = os.path.join(automated_resuts, 'final_exp_2.json')
mean_ocr_cer, mean_check_label_cer = calculate_mean_cer(json_file_path)
print(f"Mean OCR CER value: {mean_ocr_cer}")
print(f"Mean Check Label CER value: {mean_check_label_cer}")
