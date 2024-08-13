import json
import os.path
from collections import defaultdict

from src.utils.constants import results_test_trocr  # Adjust import as needed


def compare_labels(ground_truth, predicted):
    """Compare two strings and return sets of unique errors."""
    substitution_errors = defaultdict(set)
    insertion_errors = set()
    deletion_errors = set()

    len_gt = len(ground_truth)
    len_pred = len(predicted)
    max_len = max(len_gt, len_pred)

    for i in range(max_len):
        if i >= len_gt:
            insertion_errors.add(f"Insertion: '{predicted[i]}'")
        elif i >= len_pred:
            deletion_errors.add(f"Deletion: '{ground_truth[i]}'")
        elif ground_truth[i] != predicted[i]:
            substitution_errors[ground_truth[i]].add(predicted[i])

    return substitution_errors, insertion_errors, deletion_errors


def analyze_ocr_errors(json_files):
    """Analyze OCR errors from multiple JSON files."""
    substitution_errors_set = defaultdict(set)
    insertion_errors_set = set()
    deletion_errors_set = set()

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        file_name = os.path.basename(json_file)  # Extract filename from path

        print(f"Total entries in {json_file}: {len(data)}")

        for entry in data:
            if 'ground_truth_label' not in entry or 'predicted_label' not in entry:
                print(f"Entry missing required fields in {file_name}: {entry}")
                continue

            ground_truth = entry['ground_truth_label']
            predicted = entry['predicted_label']

            substitution_errors, insertion_errors, deletion_errors = compare_labels(ground_truth, predicted)

            for key, value_set in substitution_errors.items():
                substitution_errors_set[key].update(value_set)

            insertion_errors_set.update(insertion_errors)
            deletion_errors_set.update(deletion_errors)

    return substitution_errors_set, insertion_errors_set, deletion_errors_set


def print_errors(substitution_errors_set, insertion_errors_set, deletion_errors_set):
    """Print errors in a readable format."""
    print("Substitution Errors:")
    for key, value_set in substitution_errors_set.items():
        print(f"- '{key}' -> {value_set}")

    print("\nInsertion Errors:")
    for error in insertion_errors_set:
        print(f"- {error}")

    print("\nDeletion Errors:")
    for error in deletion_errors_set:
        print(f"- {error}")


def generate_mistral_7b_response(substitution_errors_set, insertion_errors_set, deletion_errors_set):
    """Generate Mistral-7b response prompt for correcting OCR errors."""
    response = "Mistral-7b Prompt:\n"
    response += "Please correct the following OCR errors:\n\n"

    response += "Substitution Errors:\n"
    for key, value_set in substitution_errors_set.items():
        response += f"- '{key}' -> {value_set}\n"

    response += "\nInsertion Errors:\n"
    for error in insertion_errors_set:
        response += f"- {error}\n"

    response += "\nDeletion Errors:\n"
    for error in deletion_errors_set:
        response += f"- {error}\n"

    return response


# Example usage:
if __name__ == "__main__":
    json_files = [
        'final_test_evaluation_results_100.json',
        'final_test_evaluation_results_75.json',
        'final_test_evaluation_results_50.json',
        'final_test_evaluation_results_25.json'
    ]
    # List of JSON files to analyze; adjust filenames as necessary

    full_paths = [os.path.join(results_test_trocr, json_file) for json_file in json_files]

    try:
        substitution_errors, insertion_errors, deletion_errors = analyze_ocr_errors(full_paths)
        # print("Errors found across all files:")
        # print_errors(substitution_errors, insertion_errors, deletion_errors)
        mistral_7b_response = generate_mistral_7b_response(substitution_errors, insertion_errors, deletion_errors)
        print(mistral_7b_response)
    except Exception as e:
        print(f"Error analyzing files: {e}")
