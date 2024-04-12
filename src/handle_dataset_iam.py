import os
import re
import math

from src.handle_dataset_washington import save_to_json
from src.utils.constants import transcription_iam_path, iam_outputs_path


class IAMLabelParser:
    """A class for parsing labels from the IAM handwriting dataset and splitting the dataset.

    Attributes:
        path (str): The file path to the IAM dataset.
        seed (int): The seed for random number generation to ensure reproducibility.
    """

    def __init__(self, path, seed=42):
        """Initializes the IAMLabelParser with a dataset path and a seed for randomization."""
        self.path = path
        self.seed = seed
        self.inside_quote = False  # Track if we're inside a quoted segment

    def parse_label(self, label):
        """Parses a single label from the IAM dataset.

        Args:
            label (str): The label string to parse.

        Returns:
            tuple: A tuple containing the image name and the processed label.
        """
        parts = label.split(" ", 8)  # Split by space to get the name and the rest of the label
        image_name = parts[0] + ".png"  # Image name is the first part
        rest_of_label = parts[-1]  # The actual label text is the last part after splitting

        # Handle |# and #| patterns to treat # as a marker for no space
        rest_of_label = re.sub(r'\|#', '', rest_of_label)  # Remove # when it follows |
        rest_of_label = re.sub(r'#\|', '', rest_of_label)  # Remove # when it precedes |
        rest_of_label = re.sub(r'#', '', rest_of_label)  # Remove #

        # Replace '|' with a space

        rest_of_label = rest_of_label.replace('|\'|', '\'')
        rest_of_label = rest_of_label.replace('|\'', '\'')
        rest_of_label = rest_of_label.replace('\'|', '\'')
        rest_of_label = rest_of_label.replace('|', ' ')

        # Ensure proper handling of punctuation
        rest_of_label = re.sub(r'\s+([,.;!?])', r'\1', rest_of_label)  # Remove spaces before commas and periods
        rest_of_label = re.sub(r'([,.;!?])', r'\1 ', rest_of_label)  # Ensure a single space after commas and periods, adjust if necessary
        rest_of_label = re.sub(r'\s*(:)\s*', r'\1 ', rest_of_label)  # Handle colon spacing

        # Adjust spaces around parentheses
        rest_of_label = re.sub(r'\(\s*', ' (', rest_of_label)  # Add space before "("
        rest_of_label = re.sub(r'\s*\)', ') ', rest_of_label)  # Add space after ")"

        # Handle quotes directly, preserving text inside as-is
        rest_of_label = self.adjust_quote_spacing(rest_of_label)

        # Directly handle quotes, attempting to preserve intended spacing
        rest_of_label, self.inside_quote = self.adjust_for_spanning_quotes(rest_of_label, self.inside_quote)

        # Normalize spaces (e.g., multiple spaces to single space, correct spacing around punctuation)
        rest_of_label = re.sub(r'\s{2,}', ' ', rest_of_label).strip()
        # print(rest_of_label)

        return image_name, rest_of_label

    def adjust_quote_spacing(self, text):
        """Adjusts spacing around quotes in text, ensuring no undesired spaces adjacent to quotes.

        Args:
            text (str): The label text potentially containing quoted segments.

        Returns:
            str: The text with adjusted spacing around quotes.
        """
        # Correctly adjust spacing around quotes
        corrected_text = re.sub(r'\s*"\s*(.*?)\s*"\s*', r' "\1" ', text)

        return corrected_text

    def adjust_for_spanning_quotes(self, text, inside_quote):
        """Adjusts text for quotes that span across labels, smartly handling the space around quotes.

        Args:
            text (str): The text of the current label.
            inside_quote (bool): Whether the previous label ended inside a quote.

        Returns:
            tuple: Updated text and the updated inside_quote state.
        """
        quote_count = text.count('"')

        # Leave labels with two quotes untouched
        if quote_count == 2:
            return text, False

        if quote_count % 3 == 0 and quote_count != 0:
            if not inside_quote:
                quote_index = text.rfind('"')
                text = text[:quote_index].rstrip() + ' ' + '"' + text[quote_index + 1:].lstrip()
                inside_quote = True
            return text, inside_quote

        if quote_count == 1:  # Handling a single quote
            if inside_quote:  # If we're currently inside a quote, this label ends it
                if not text.endswith('"'):  # If the quote is not at the end, adjust space after the quote
                    quote_index = text.rfind('"')
                    text = text[:quote_index].rstrip() + '"' + ' ' + text[quote_index + 1:].lstrip()
                inside_quote = False

            else:  # This label starts a quote
                if not text.startswith('"'):  # If the quote is not at the beginning, adjust space before the quote
                    quote_index = text.find('"')
                    text = text[:quote_index].rstrip() + ' ' + '"' + text[quote_index + 1:].lstrip()
                else:  # If the quote is at the beginning, ensure it's correctly positioned
                    text = '"' + text[1:].lstrip()
                inside_quote = True
            print(f"text with quote: {text}, inside quote: {inside_quote}")
        return text, inside_quote

    def get_sequential_subsets(self, training_pct, validation_pct):
        """Sequentially splits the dataset into training, validation, and testing subsets.

        Args:
            training_pct (int): The percentage of the dataset to allocate to the training set.
            validation_pct (int): The percentage of the dataset to allocate to the validation set.

        Returns:
            tuple: A tuple of dictionaries for the training, validation, and testing sets.
        """
        with open(self.path, 'r') as file:
            lines = file.readlines()

        total = len(lines)
        training_size = math.ceil(total * training_pct / 100)
        validation_size = math.ceil(total * validation_pct / 100)

        training_lines = lines[:training_size]
        validation_lines = lines[training_size:training_size + validation_size]
        testing_lines = lines[training_size + validation_size:]

        return self._lines_to_dict(training_lines), self._lines_to_dict(validation_lines), self._lines_to_dict(
            testing_lines)

    def _lines_to_dict(self, lines):
        """Converts lines to a dictionary with image names as keys and labels as values."""
        dict_images_labels = {}
        for line in lines:
            name, label = self.parse_label(line)
            dict_images_labels[name] = label
        return dict_images_labels


iam_parser = IAMLabelParser(transcription_iam_path)
training_data, validation_data, testing_data = iam_parser.get_sequential_subsets(80, 10)
save_to_json(training_data, os.path.join(iam_outputs_path, 'train', 'training_seq_data.json'))
save_to_json(validation_data, os.path.join(iam_outputs_path, 'valid', 'validation_seq_data.json'))
save_to_json(testing_data, os.path.join(iam_outputs_path, 'test', 'testing_seq_data.json'))
