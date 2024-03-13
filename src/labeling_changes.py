import json
import os


def calculate_label_change_percentages(directory, file_name):
    """
    Calculate percentages of label changes based on CER values from a specified JSON file, formatted to two decimal places, and return them.

    Args:
        directory (str): The directory where the JSON file is located.
        file_name (str): The name of the JSON file.

    Returns:
        dict: A dictionary containing the percentages of label changes for CER=0 and CER>0, formatted to two decimal places.
    """
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'r') as file:
        json_data = json.load(file)

    same_label_cer_0_count = 0
    modified_label_cer_0_count = 0
    same_label_cer_greater_than_0_count = 0
    modified_label_cer_greater_than_0_count = 0

    for item in json_data:
        ocr_cer = item["OCR"]["cer"]
        ocr_label = item["OCR"]["predicted_label"]
        mistral_label = item["MISTRAL"]["predicted_label"]
        label_changed = (ocr_label != mistral_label)

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

    total_cer_0 = same_label_cer_0_count + modified_label_cer_0_count
    total_cer_greater_than_0 = same_label_cer_greater_than_0_count + modified_label_cer_greater_than_0_count

    # Calculate and format percentages
    same_label_cer_0_percentage = round((same_label_cer_0_count / total_cer_0 * 100) if total_cer_0 else 0, 2)
    modified_label_cer_0_percentage = round((modified_label_cer_0_count / total_cer_0 * 100) if total_cer_0 else 0, 2)
    same_label_cer_greater_than_0_percentage = round((same_label_cer_greater_than_0_count / total_cer_greater_than_0 * 100) if total_cer_greater_than_0 else 0, 2)
    modified_label_cer_greater_than_0_percentage = round((modified_label_cer_greater_than_0_count / total_cer_greater_than_0 * 100) if total_cer_greater_than_0 else 0, 2)

    return {
        "same_label_cer_0_percentage": same_label_cer_0_percentage,
        "modified_label_cer_0_percentage": modified_label_cer_0_percentage,
        "same_label_cer_greater_than_0_percentage": same_label_cer_greater_than_0_percentage,
        "modified_label_cer_greater_than_0_percentage": modified_label_cer_greater_than_0_percentage
    }

# Example usage:
# percentages = calculate_label_change_percentages('path/to/directory', 'final_test_evaluation_from_mistral_50_50.json')
# print(percentages)
