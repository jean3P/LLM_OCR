import os
import torch
import gc
from src.Mix_Mistral_TrainSet import extract_and_combine
from src.TrOCR import TrainingConfig
from src.TrOCREvaluation import train_and_save_model, calculate_sample_size
from src.handle_dataset import load_from_json
from src.main_mixtral import evaluate_test_data_mixtral7B
from src.test import evaluate_test_data, create_and_save_subset_from, create_and_save_subset_from_for_train
from src.utils.constants import outputs_path, automated_resuts, results_test_trocr, TOKEN


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


def automate_workflow(start_percentage=25, increments=25, max_iterations=3, training_data_path='', valid_data_path='',
                      training_config=TrainingConfig):
    # mistral_model_name = "mistralai/Mixtral-8x7B-v0.1"
    mistral_model_name = "mistralai/Mistral-7B-v0.1"

    for iteration in range(max_iterations):
        name_file_tested = f"test_evaluation_results_{start_percentage}.json"
        # 1. Train TrOCR and save models
        print(f"=== Train TrOCR - {start_percentage} ===")
        model_name = f"trained_trocr_model_seq_{start_percentage}"
        processor_name = f"trocr_processor_seq_{start_percentage}"
        model_save_dir = os.path.join(outputs_path, 'model', model_name)
        processor_save_dir = os.path.join(outputs_path, 'model', processor_name)

        # 1.2 Calculate the new subset for training
        last_used_line = calculate_sample_size(training_data_path, start_percentage)
        first_name_test = f"test_from_train_{start_percentage}.json"
        test_name_first = os.path.join(outputs_path, 'test', first_name_test)
        if not file_exists(test_name_first, outputs_path):
            create_and_save_subset_from_for_train(training_data_path, last_used_line, test_name_first)

        name_test = f"test_from_train_{start_percentage}_{100 - start_percentage}.json"
        test_name = os.path.join(outputs_path, 'test', name_test)
        if not file_exists(name_test, outputs_path) and start_percentage < 100:
            create_and_save_subset_from(training_data_path, last_used_line + 1, test_name)

        # Run TrOCR training
        print(f"=== TRAINING - MODEL - {start_percentage} ===")
        clear_cuda_cache()
        if not directory_exists(model_save_dir) and not directory_exists(processor_save_dir):
            train_and_save_model(training_data_path, valid_data_path, model_save_dir, processor_save_dir,
                                 training_config, start_percentage)
            clear_cuda_cache()

        # 2. Evaluate final test dataset with TrOCR
        print(f"=== EVALUATE TrOCR WITH FINAL TEST - {start_percentage} ===")
        name_file_tested_final = f"final_test_evaluation_results_{start_percentage}.json"
        test_final = os.path.join(outputs_path, 'test', 'testing_seq_data.json')
        if not file_exists(name_file_tested_final, results_test_trocr):
            evaluate_test_data(processor_save_dir, model_save_dir, test_final, name_file_tested_final)

        # 3. MISTRAL
        # Mistral for final test
        print(f"=== MISTRAL WITHOUT SELF TRAINING WITH FINAL TEST - {start_percentage} ===")
        name_mistral_1_final = f"final_test_evaluation_from_mistral_{start_percentage}.json"
        results_path_from_ocr_final = os.path.join(results_test_trocr, name_file_tested_final)
        clear_cuda_cache()
        if not file_exists(name_mistral_1_final, automated_resuts):
            loaded_data = load_from_json(results_path_from_ocr_final)
            evaluate_test_data_mixtral7B(loaded_data, name_mistral_1_final, True)
            clear_cuda_cache()

        if start_percentage < 100:
            # Evaluation Self Training
            print(
                f"=== EVALUATE TrOCR SELF TRAINING - {start_percentage} WITH REMAINING TEST {100 - start_percentage} ===")
            if not file_exists(name_file_tested, results_test_trocr):
                evaluate_test_data(processor_save_dir, model_save_dir, test_name, name_file_tested)

            # Mistral for self-training
            name_mistral_1 = f"test_evaluation_from_mistral_{start_percentage}.json"
            results_path_from_ocr = os.path.join(results_test_trocr, name_file_tested)
            clear_cuda_cache()
            if not file_exists(name_mistral_1, automated_resuts):
                loaded_data = load_from_json(results_path_from_ocr)
                print(f"=== MISTRAL SELF TRAINING - {start_percentage} ===")
                evaluate_test_data_mixtral7B(loaded_data, name_mistral_1, False)
                clear_cuda_cache()
            print(f"=== MERGE - {start_percentage}-{100 - start_percentage} ===")
            # 4. Merge datasets for new training set
            name_mixed_file = f"mixed_train_seq_{start_percentage}_{100 - start_percentage}.json"
            test_mistral = os.path.join(automated_resuts, name_mistral_1)
            extract_and_combine(test_mistral, test_name_first, name_mixed_file)

            # 5. Merge datasets for new training set
            print(f"=== TRAINING TrOCR SELF - {start_percentage}-{100 - start_percentage} ===")
            training_data_path_2 = os.path.join(automated_resuts, name_mixed_file)
            model_name_2 = f"trained_trocr_model_seq_{start_percentage}_{100 - start_percentage}"
            processor_name_2 = f"trocr_processor_seq_{start_percentage}_{100 - start_percentage}"
            model_save_dir_2 = os.path.join(outputs_path, 'model', model_name_2)
            processor_save_dir_2 = os.path.join(outputs_path, 'model', processor_name_2)
            clear_cuda_cache()
            if not directory_exists(model_save_dir_2) and not directory_exists(processor_save_dir_2):
                train_and_save_model(training_data_path_2, valid_data_path, model_save_dir_2,
                                     processor_save_dir_2,
                                     training_config, 100)
                clear_cuda_cache()

            print(f"=== TEST OCR MODEL SELF - {start_percentage}-{100 - start_percentage} ===")
            name_file_tested_final = f"final_test_evaluation_results_{start_percentage}_{100 - start_percentage}.json"
            if not file_exists(name_file_tested_final, results_test_trocr):
                evaluate_test_data(processor_save_dir_2, model_save_dir_2, test_final, name_file_tested_final)

            # 7. MISTRAL
            print(f"=== FINAL TEST MISTRAL - {start_percentage}-{100 - start_percentage} ===")
            name_mistral_3 = f"final_test_evaluation_from_mistral_{start_percentage}_{100 - start_percentage}.json"
            results_path_from_ocr = os.path.join(results_test_trocr, name_file_tested_final)
            clear_cuda_cache()
            if not file_exists(name_mistral_3, automated_resuts):
                loaded_data = load_from_json(results_path_from_ocr)
                evaluate_test_data_mixtral7B(loaded_data, name_mistral_3, True)
                clear_cuda_cache()

        start_percentage = start_percentage + increments


if __name__ == "__main__":
    training = os.path.join(outputs_path, 'train', 'training_seq_data.json')
    valid = os.path.join(outputs_path, 'valid', 'validation_seq_data.json')
    training_config = TrainingConfig(BATCH_SIZE=10, EPOCHS=35, LEARNING_RATE=0.00005)
    automate_workflow(25, 25, 4, training, valid)
