import json
import os
from src.utils.constants import outputs_path


def get_names_without_extension(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    names = [name.replace('.png', '') for name in data.keys()]
    return names


# Sample usage
if __name__ == "__main__":
    json_file_path = os.path.join(outputs_path, 'valid', 'validation_seq_data.json')
    # json_file_path = os.path.join(outputs_path, 'train', 'training_seq_data.json')
    # json_file_path = os.path.join(outputs_path, 'test', 'testing_seq_data.json')
    names_without_extension = get_names_without_extension(json_file_path)

    for name in names_without_extension:
        print(name)
