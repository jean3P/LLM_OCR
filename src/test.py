import os

import evaluate
import torch
from PIL import Image

from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from customOCRDataset import DatasetConfig
from data_frame_handler import DataFrameHandler
from handle_dataset import save_to_json, load_from_json
from utils.constants import outputs_path, model_save_path, processor_save_path, results_test_trocr

device = torch.device('mps')

# Load the saved model
model = VisionEncoderDecoderModel.from_pretrained(model_save_path)

# Load the saved processor
processor = TrOCRProcessor.from_pretrained(processor_save_path)

# Move the model to the appropriate device
model.to(device)

handler = DataFrameHandler()
test = load_from_json(os.path.join(outputs_path, 'test', 'testing_data.json'))
test_df = handler.dict_to_dataframe(test)

# Initialize CER metric
cer_metric = evaluate.load('cer')


def ocr(image, processor, model):
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


def evaluate_test_data(df, processor, model):
    """
    Evaluate the model on the entire test dataset and save results in a JSON file.

    Args:
        df (pd.DataFrame): The DataFrame containing test data.
        processor (TrOCRProcessor): The processor for the TrOCR model.
        model (VisionEncoderDecoderModel): The trained TrOCR model.
    """
    results = []
    for i, row in df.iterrows():
        image_path = os.path.join(DatasetConfig.DATA_ROOT, row['file_name'])
        image = Image.open(image_path).convert('RGB')
        predicted_text = ocr(image, processor, model)

        # Calculate CER
        cer = cer_metric.compute(predictions=[predicted_text], references=[row['text']])

        results.append({
            'file_name': row['file_name'],
            'actual_label': row['text'],
            'predicted_label': predicted_text,
            'cer': cer
        })

    path_file = os.path.join(results_test_trocr, 'test_evaluation_results.json')
    save_to_json(results, path_file)


# Example usage
evaluate_test_data(test_df, processor, model)
print("The data of test is saved.")
