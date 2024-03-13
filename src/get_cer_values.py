import json
import os


def calculate_cer_values(directory_path_mistral, filename_mistral, path_directory_ocr, other_filename_ocr):
    """
    Calculates and returns the mean CER values for a specified Mistral file and an OCR file.

    Args:
        directory_path_mistral (str): The directory where the Mistral JSON file is located.
        filename_mistral (str): The name of the Mistral JSON file.
        path_directory_ocr (str): The directory where the OCR JSON file is located.
        other_filename_ocr (str): The name of the OCR JSON file.

    Returns:
        dict: A dictionary with mean CER values for both Mistral and OCR files, rounded to two decimal places.
    """
    # Load and calculate mean CER for the Mistral file
    mistral_file_path = os.path.join(directory_path_mistral, filename_mistral)
    mistral_data = load_from_json(mistral_file_path)
    mean_cer_mistral = calculate_mean(mistral_data, 'MISTRAL', 'cer')

    # Load and calculate mean CER for the OCR file
    ocr_file_path = os.path.join(path_directory_ocr, other_filename_ocr)
    ocr_data = load_from_json(ocr_file_path)
    # Calculate the total CER and total confidence
    total_cer = sum(item['cer'] for item in ocr_data)

    # Calculate the number of items
    num_items = len(ocr_data)

    # Calculate the mean CER and mean confidence
    mean_cer_ocr = total_cer / num_items
    results = {
        'Mean CER OCR': round(mean_cer_ocr * 100, 2) if mean_cer_ocr is not None else None,
        'Mean CER Mistral': round(mean_cer_mistral*100, 2) if mean_cer_mistral is not None else None,
    }

    return results


# Function to load JSON data from a file
def load_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to calculate mean values given data, a key, and a field
def calculate_mean(data, key, field):
    values = [item[key][field] for item in data if key in item and field in item[key]]
    return sum(values) / len(values) if values else None

# Example usage
# directory_path = 'path_to_mistral_data'
# filename = 'mistral_file.json'
# path_directory = 'path_to_ocr_data'
# other_filename = 'ocr_file.json'
# results = calculate_cer_values(directory_path, filename, path_directory, other_filename)
# print(results)
