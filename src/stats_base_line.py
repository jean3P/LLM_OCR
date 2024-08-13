import json
import os

from src.utils.constants import outputs_path

from src.utils.constants import automated_resuts
# Function to calculate stats
def calculate_stats(data):
    total_ocr_cer = 0
    total_prompt_correcting_cer = 0
    total_confidence = 0
    # yes_count = 0
    # no_count = 0
    total_entries = len(data)

    for entry in data:
        total_ocr_cer += entry["OCR"]["cer"]
        total_prompt_correcting_cer += entry["Prompt correcting"]["cer"]
        total_confidence += entry["Prompt correcting"]["confidence"]
        # if entry["Prompt checking"]["response"] == "Yes":
        #     yes_count += 1
        # elif entry["Prompt checking"]["response"] == "No":
        #     no_count += 1

    mean_ocr_cer = round((total_ocr_cer / total_entries), 3)  # converting to percentage
    mean_prompt_correcting_cer = round((total_prompt_correcting_cer / total_entries), 3)  # converting to percentage
    mean_confidence = total_confidence / total_entries  # already in percentage

    stats = {
        "mean_ocr_cer_percentage": mean_ocr_cer,
        "mean_prompt_correcting_cer_percentage": mean_prompt_correcting_cer,
        "mean_confidence_percentage": mean_confidence,
        # "yes_count": yes_count,
        # "no_count": no_count
    }

    return stats


# Load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Path to your JSON file
# json_file_path = os.path.join(outputs_path, 'base_line_with_check_and_correct_ocr_v4', 'automated_results',
#                               'final_test_evaluation_from_mistral_100.json')

json_file_path = os.path.join(automated_resuts, 'final_test_evaluation_from_mistral_25.json')

# Load the data
data = load_json_file(json_file_path)

# Calculate stats
stats = calculate_stats(data)

# Print stats
print(json.dumps(stats, indent=4))
