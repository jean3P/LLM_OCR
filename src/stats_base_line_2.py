import json
import os

from src.utils.constants import automated_resuts


# Function to calculate stats
def calculate_stats(data):
    total_ocr_cer = 0
    total_prompt_correcting_cer = 0
    total_confidence = 0
    ocr_yes_count = 0
    ocr_no_count = 0
    grammar_yes_count = 0
    grammar_no_count = 0
    total_entries = len(data)

    for entry in data:
        total_ocr_cer += entry["OCR"]["cer"]
        total_prompt_correcting_cer += entry["Prompt correcting: OCR checking"]["cer"]
        total_confidence += entry["Prompt correcting: OCR checking"]["confidence"]

        if entry["Prompt checking: OCR checking"]["response"] == "Yes":
            ocr_yes_count += 1
        elif entry["Prompt checking: OCR checking"]["response"] == "No":
            ocr_no_count += 1

        if entry["Prompt checkin: Grammar checking"]["response"] == "Yes":
            grammar_yes_count += 1
        elif entry["Prompt checkin: Grammar checking"]["response"] == "No":
            grammar_no_count += 1

    mean_ocr_cer = round((total_ocr_cer / total_entries), 3)  # converting to percentage
    mean_prompt_correcting_cer = round((total_prompt_correcting_cer / total_entries),
                                       3)  # converting to percentage
    mean_confidence = total_confidence / total_entries  # already in percentage

    stats = {
        "mean_ocr_cer_percentage": mean_ocr_cer,
        "mean_prompt_correcting_cer_percentage": mean_prompt_correcting_cer,
        "mean_confidence_percentage": mean_confidence,
        "ocr_yes_count": ocr_yes_count,
        "ocr_no_count": ocr_no_count,
        "grammar_yes_count": grammar_yes_count,
        "grammar_no_count": grammar_no_count
    }

    return stats


# Load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Path to your JSON file
json_file_path = os.path.join(automated_resuts, 'final_test_evaluation_from_mistral_25.json')

# Load the data
data = load_json_file(json_file_path)

# Calculate stats
stats = calculate_stats(data)

# Print stats
print(json.dumps(stats, indent=4))
