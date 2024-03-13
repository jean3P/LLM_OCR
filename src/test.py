import os

import evaluate
import torch
from PIL import Image

from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from confidence_calculator import calculate_confidence
from customOCRDataset import DatasetConfig
from data_frame_handler import DataFrameHandler
from handle_dataset import save_to_json, load_from_json
from utils.constants import results_test_trocr, outputs_path_test

#
# # Load the saved model
#
#
# # Move the model to the appropriate device
# model.to(device)

# handler = DataFrameHandler()
# test = load_from_json(os.path.join(outputs_path, 'test', 'testing_seq_data.json'))
# test_df = handler.dict_to_dataframe(test)
#
# # Initialize CER metric
# cer_metric = evaluate.load('cer')


def ocr(image, processor, model, device):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.


    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def create_and_save_subset_from(training_data_path, start_position, output_file_name):
    """
    Creates a new training set from the specified start position to the end of the original training set and saves it to JSON.

    Args:
        training_data_path (str): Path to the original training data JSON file.
        start_position (int): The starting position (0-based index) for the new subset.
        output_file_name (str): Name for the output JSON file containing the new subset.
    """
    # Load the original training data
    original_data = load_from_json(training_data_path)

    # Convert the dictionary to a list of tuples to preserve ordering
    items_list = list(original_data.items())
    # Create a new subset starting from the given position
    subset_list = items_list[start_position:]
    # print(subset_list)

    # Convert the subset list back to a dictionary
    subset_dict = dict(subset_list)
    # Define the path for the output file
    # output_path = os.path.join(output_file_name)

    # Save the new subset to JSON
    save_to_json(subset_dict, output_file_name)

    print(f"New training set saved to {output_file_name}")


def create_and_save_subset_from_for_train(training_data_path, last_position, output_file_name):
    """
    Creates a new training set from the specified start position to the end of the original training set and saves it to JSON.

    Args:
        training_data_path (str): Path to the original training data JSON file.
        start_position (int): The starting position (0-based index) for the new subset.
        output_file_name (str): Name for the output JSON file containing the new subset.
    """
    # Load the original training data
    original_data = load_from_json(training_data_path)

    # Convert the dictionary to a list of tuples to preserve ordering
    items_list = list(original_data.items())
    # Create a new subset starting from the given position
    subset_list = items_list[:last_position]
    # print(subset_list)

    # Convert the subset list back to a dictionary
    subset_dict = dict(subset_list)
    # Define the path for the output file
    # output_path = os.path.join(output_file_name)

    # Save the new subset to JSON
    save_to_json(subset_dict, output_file_name)

    print(f"New training set saved to {output_file_name}")


def evaluate_test_data(processor, model, test_name, name_file_tested):
    """
    Evaluate the model on the entire test dataset and save results in a JSON file.

    Args:
        df (pd.DataFrame): The DataFrame containing test data.
        processor (TrOCRProcessor): The processor for the TrOCR model.
        model (VisionEncoderDecoderModel): The trained TrOCR model.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model = VisionEncoderDecoderModel.from_pretrained(model)
    model.to(device)
    cer_metric = evaluate.load('cer')
    handler = DataFrameHandler()
    processor = TrOCRProcessor.from_pretrained(processor)
    test = load_from_json(test_name)
    test_df = handler.dict_to_dataframe(test)
    results = []
    for i, row in test_df.iterrows():
        image_path = os.path.join(DatasetConfig.DATA_ROOT, row['file_name'])
        image = Image.open(image_path).convert('RGB')
        predicted_text = ocr(image, processor, model, device)

        # Calculate CER
        cer = cer_metric.compute(predictions=[predicted_text], references=[row['text']])

        # Calculate confidence scores
        confidence_score = calculate_confidence(row['text'], predicted_text)

        results.append({
            'file_name': row['file_name'],
            'ground_truth_label': row['text'],
            'predicted_label': predicted_text,
            'confidence': confidence_score,
            'cer': cer
        })

    path_file = os.path.join(results_test_trocr, name_file_tested)
    save_to_json(results, path_file)

# # Example usage
# evaluate_test_data(test_df, processor, model)
# print("The data of test is saved.")
