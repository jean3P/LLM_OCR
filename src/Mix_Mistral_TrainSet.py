import json
import os

from src.handle_dataset import save_to_json, load_from_json
from src.utils.constants import results_LLM_mistral_3, outputs_path, results_mixed_LLM_MISTRAL, automated_resuts


def extract_and_combine(file_path_1, file_path_2, name_file=''):
    # Ensure the directory path is correctly spelled and passed as an argument to the function
    # Load and process the first JSON file
    full_path_1 = file_path_1
    with open(full_path_1, 'r') as file:
        data_1 = json.load(file)
    # Assuming data_1 structure is a list of dicts with 'file_name' and 'MISTRAL' keys
    mistral_predictions = {item['file_name']: item['MISTRAL']['predicted_label'] for item in data_1}

    # Load the second JSON file
    full_path_2 = file_path_2
    with open(full_path_2, 'r') as file:
        data_2 = json.load(file)

    # Combine both dictionaries. This assumes data_2 is already a dictionary that can be directly combined
    combined_data = {**data_2, **mistral_predictions}

    # Save the combined data to a new file in the same directory
    output_path = os.path.join(automated_resuts, name_file)
    with open(output_path, 'w') as file:
        json.dump(combined_data, file, indent=4)

    print(f"Combined data saved to {output_path}")

    # return combined_data

    # # Write the combined data to a new JSON file
    # with open(output_file_path, 'w') as file:
    #     json.dump(combined_data, file, indent=4)


# save_mistral_output = os.path.join(automated_resuts, 'test_evaluation_from_mistral_25.json')
# training = os.path.join(outputs_path, 'train', 'training_seq_data.json')
#
# extract_and_combine(save_mistral_output, training, 'mixed_train_seq_75_25.json')
# # results = (os.path.join(outputs_path, 'mixed', 'train', 'mixed_train_seq_75_25.json'))
# # save_to_json(new_file, results)
# # print(new_file)
