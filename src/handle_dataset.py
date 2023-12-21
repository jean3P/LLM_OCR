import json
import os
import re
import random
import math

from utils.constants import REPLACEMENTS, transcription_path, outputs_path


def save_to_json(dict_data, filename):
    """Saves a dictionary to a JSON file.
    Args:
        dict_data (dict): The dictionary to save.
        filename (str): The path to the file where the dictionary will be saved.
    """
    with open(filename, 'w') as file:
        json.dump(dict_data, file, indent=4)


def load_from_json(filename):
    """Loads data from a JSON file into a dictionary.
    Args:
        filename (str): The path to the JSON file to be loaded.
    Returns:
        dict: The dictionary containing the data loaded from the JSON file.
    """
    with open(filename, 'r') as file:
        return json.load(file)


class LabelParser:
    """A class for parsing labels from a dataset and splitting the dataset.

    Attributes:
        path (str): The file path to the dataset.
        seed (int): The seed for random number generation to ensure reproducibility.
        pattern (re.Pattern): Compiled regular expression pattern for replacements.
    """
    def __init__(self, path, seed=42):
        """Initializes the LabelParser with a dataset path and a seed for randomization."""
        self.path = path
        self.seed = seed
        self.pattern = re.compile('|'.join(re.escape(key) for key in REPLACEMENTS.keys()))

    def parse_label(self, label):
        """Parses a single label from the dataset.
        Args:
            label (str): The label string to parse.
        Returns:
            tuple: A tuple containing the image name and the processed label.
        """
        image_name, rest_of_label = label[:6], label[6:]
        rest_of_label = self.pattern.sub(lambda x: REPLACEMENTS[x.group()], rest_of_label)
        rest_of_label = rest_of_label.rstrip('\n')

        # Remove the first space character if it exists
        if rest_of_label.startswith(' '):
            rest_of_label = rest_of_label[1:]

        return image_name, rest_of_label

    def get_subsets(self, training_pct, validation_pct):
        """Splits the dataset into training, validation, and testing subsets.
        Args:
            training_pct (int): The percentage of the dataset to allocate to the training set.
            validation_pct (int): The percentage of the dataset to allocate to the validation set.
        Returns:
            tuple: A tuple of dictionaries for the training, validation, and testing sets.
        """
        with open(self.path, 'r') as file:
            lines = file.readlines()

        random.seed(self.seed)
        random.shuffle(lines)

        total = len(lines)
        training_size = math.ceil(total * training_pct / 100)
        validation_size = math.ceil(total * validation_pct / 100)

        training_lines = lines[:training_size]
        validation_lines = lines[training_size:training_size + validation_size]
        testing_lines = lines[training_size + validation_size:]

        return self._lines_to_dict(training_lines), self._lines_to_dict(validation_lines), self._lines_to_dict(
            testing_lines)

    def _lines_to_dict(self, lines):
        dict_images_labels = {}
        for line in lines:
            name, label = self.parse_label(line)
            name_ext = name + '.png'
            dict_images_labels[name_ext] = label
        return dict_images_labels


# Initialize LabelParser
label_parser = LabelParser(transcription_path)

# Split dataset and save to JSON files
training_data, validation_data, testing_data = label_parser.get_subsets(70, 15)

save_to_json(training_data, os.path.join(outputs_path, 'train', 'training_data.json'))
save_to_json(validation_data, os.path.join(outputs_path, 'valid', 'validation_data.json'))
save_to_json(testing_data, os.path.join(outputs_path, 'test', 'testing_data.json'))
