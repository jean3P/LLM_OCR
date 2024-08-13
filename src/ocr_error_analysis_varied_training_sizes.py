import os
import torch
from transformers import BitsAndBytesConfig
from src.TrOCR import TrainingConfig
from src.TrOCREvaluation import train_and_save_model, calculate_sample_size
from src.handle_dataset_washington import load_from_json
from src.test import evaluate_test_data, create_and_save_subset_from, create_and_save_subset_from_for_train
from src.utils.constants import outputs_path, automated_resuts, results_test_trocr
from src.utils.model_utils import load_mistral_model, create_pipeline, evaluate_mistral_model
from src.utils.utils import directory_exists, file_exists, clear_cuda_cache
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('workflow_logger', 'workflow.log')


def automate_workflow(start_percentage=25, increments=25, max_iterations=3, training_data_path='', valid_data_path='',
                      training_config=TrainingConfig):
    mistral_model_name = "mistralai/Mistral-7B-v0.1"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    for iteration in range(max_iterations):
        logger.info(f"Starting iteration {iteration + 1} with start percentage {start_percentage}")

        # 1. Train TrOCR and save models
        logger.info(f"=== Train TrOCR - {start_percentage} ===")
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
            logger.info(f"Created subset for training: {first_name_test}")

        name_test = f"test_from_train_{start_percentage}_{100 - start_percentage}.json"
        test_name = os.path.join(outputs_path, 'test', name_test)
        if not file_exists(name_test, outputs_path) and start_percentage < 100:
            create_and_save_subset_from(training_data_path, last_used_line + 1, test_name)
            logger.info(f"Created subset for testing: {name_test}")

        # Run TrOCR training
        logger.info(f"=== TRAINING - MODEL - {start_percentage} ===")
        if not directory_exists(model_save_dir) and not directory_exists(processor_save_dir):
            train_and_save_model(training_data_path, valid_data_path, model_save_dir, processor_save_dir,
                                 training_config, start_percentage)
            logger.info(f"Model trained and saved at {model_save_dir}")
            clear_cuda_cache()

        # 2. Evaluate final test dataset with TrOCR
        logger.info(f"=== EVALUATE TrOCR WITH FINAL TEST - {start_percentage} ===")
        name_file_tested_final = f"final_test_evaluation_results_{start_percentage}.json"
        test_final = os.path.join(outputs_path, 'test', 'testing_seq_data.json')
        if not file_exists(name_file_tested_final, results_test_trocr):
            evaluate_test_data(processor_save_dir, model_save_dir, test_final, name_file_tested_final)
            logger.info(f"Evaluation results saved at {name_file_tested_final}")

        # 3. MISTRAL
        # Mistral for final test
        logger.info(f"=== MISTRAL WITHOUT SELF TRAINING WITH FINAL TEST - {start_percentage} ===")
        name_mistral_1_final = f"final_test_evaluation_from_mistral_{start_percentage}.json"
        results_path_from_ocr_final = os.path.join(results_test_trocr, name_file_tested_final)
        if not file_exists(name_mistral_1_final, automated_resuts):
            loaded_data = load_from_json(results_path_from_ocr_final)
            with torch.no_grad():
                mistral_model, mistral_tokenizer = load_mistral_model(mistral_model_name, quantization_config)
                mistral_pipe = create_pipeline(mistral_model, mistral_tokenizer)
                evaluate_mistral_model(loaded_data, training_data_path, mistral_pipe, name_mistral_1_final, mistral_tokenizer)
                logger.info(f"Mistral model evaluated and results saved at {name_mistral_1_final}")
            del mistral_model
            del mistral_tokenizer
            del mistral_pipe
            clear_cuda_cache()

        start_percentage = start_percentage + increments
        logger.info(f"Completed iteration {iteration + 1}, updated start percentage to {start_percentage}")


if __name__ == "__main__":
    training = os.path.join(outputs_path, 'train', 'training_seq_data.json')
    valid = os.path.join(outputs_path, 'valid', 'validation_seq_data.json')
    training_config = TrainingConfig(BATCH_SIZE=10, EPOCHS=35, LEARNING_RATE=0.00005)
    automate_workflow(25, 25, 4, training, valid)
