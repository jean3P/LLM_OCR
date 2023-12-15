import os
from dataclasses import dataclass

import pandas as pd
from transformers import TrOCRProcessor

from customOCRDataset import ModelConfig, CustomOCRDataset, DatasetConfig
from handle_dataset import load_from_json
from utils.constants import outputs_path


class DataFrameHandler:
    """
    A class to handle operations related to converting dictionaries to pandas DataFrames
    and sampling a specific percentage of data.
    """

    def __init__(self):
        """
        Initializes the DataFrameHandler with specified column names.

        Args:
            column_names (list): A list of strings representing the column names for the DataFrame.
        """
        self.column_names = ['file_name', 'text']

    def dict_to_dataframe(self, data_dict):
        """
        Converts a dictionary to a pandas DataFrame with specified column names.

        Args:
            data_dict (dict): The dictionary to be converted to a DataFrame.
            column_names (list): A list of strings representing the column names for the DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame created from the input dictionary.
        """
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.reset_index(inplace=True)
        df.columns = ['file_name', 'text']
        return df

    def sample_dataframe(self, df, percentage):
        """
        Samples a given percentage of rows from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to sample from.
            percentage (float): The percentage of rows to sample (between 0 and 100).

        Returns:
            pd.DataFrame: A sampled subset of the original DataFrame.
        """
        return df.sample(frac=percentage / 100)


training = load_from_json(os.path.join(outputs_path, 'train', 'training_data.json'))
valid = load_from_json(os.path.join(outputs_path, 'valid', 'validation_data.json'))
test = load_from_json(os.path.join(outputs_path, 'test', 'testing_data.json'))

handler = DataFrameHandler()

train_df = handler.dict_to_dataframe(training)
valid_df = handler.dict_to_dataframe(valid)
test_df = handler.dict_to_dataframe(test)

sampled_train_df = handler.sample_dataframe(train_df, 10)


@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 48
    EPOCHS: int = 35
    LEARNING_RATE: float = 0.00005


# """
# Initializing the processor for data preprocessing and tokenization. Creating datasets for training and validation.
# """
#
# processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
# train_dataset = CustomOCRDataset(
#     root_dir=os.path.join(DatasetConfig.DATA_ROOT, 'train/'),
#     df=train_df,
#     processor=processor
# )
# valid_dataset = CustomOCRDataset(
#     root_dir=os.path.join(DatasetConfig.DATA_ROOT, 'valid/'),
#     df=valid_df,
#     processor=processor
# )
#
# print("Number of training examples:", len(train_dataset))
# print("Number of validation examples:", len(valid_dataset))