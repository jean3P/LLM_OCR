import torch
import gc
import os


def clear_cuda_cache():
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()


def file_exists(file_name, path):
    # Join the path and file name to get the full file path
    file_path = os.path.join(path, file_name)
    # Check if the file exists and return the result
    return os.path.isfile(file_path)


def directory_exists(directory_path):
    """
    Check if the specified directory exists.

    Args:
        directory_path (str): The path to the directory to check.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return os.path.isdir(directory_path)
