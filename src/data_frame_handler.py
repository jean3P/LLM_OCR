import pandas as pd


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
