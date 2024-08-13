# from src.utils.constants import automated_resuts, results_test_trocr
#
# percentage_train_set = 75
#
# # Directory file is results_test_trocr
# self_training_file_name_ocr_75 = "test_evaluation_results_75.json"
# final_test_file_name_ocr_75_25 = "final_test_evaluation_results_75_25.json"
# final_test_file_name_ocr_75 = "final_test_evaluation_results_75.json"
#
# # Directory file is automated_resuts
# self_training_file_name_mistral_75 = "test_evaluation_from_mistral_75.json"
# final_test_file_name_mistral_75 = "final_test_evaluation_from_mistral_75.json"
# final_test_file_name_mistral_75_25 = "final_test_evaluation_from_mistral_75_25.json"
from src.get_cer_values import calculate_cer_values
from src.labeling_changes import calculate_label_change_percentages


def format_or_na(value):
    """Format the value with two decimal places or return 'N/A' if None."""
    return "{:.2f}%".format(value) if value is not None else "N/A"


def generate_latex_table(automated_results_dir, results_test_trocr_dir,
                         self_training_file_name_ocr_75=None, final_test_file_name_ocr_75_25=None,
                         final_test_file_name_ocr_75=None,
                         self_training_file_name_mistral_75=None, final_test_file_name_mistral_75=None,
                         final_test_file_name_mistral_75_25=None):
    """
    Generates a LaTeX table with CER values and label change percentages for OCR and Mistral evaluations.

    Args:
        automated_results_dir (str): Directory containing Mistral JSON files.
        results_test_trocr_dir (str): Directory containing OCR JSON files.
        self_training_ocr (str): OCR JSON file name for self-training.
        final_test_ocr_25 (str): OCR JSON file name for 25% final test.
        final_test_ocr_75 (str): OCR JSON file name for 75% final test.
        self_training_mistral (str): Mistral JSON file name for self-training.
        final_test_mistral_75 (str): Mistral JSON file name for 75% final test.
        final_test_mistral_75_25 (str): Mistral JSON file name for 75%+25% final test.

    Returns:
        dict: Dictionary containing CER values and label change percentages.
    """

    # Function to calculate CER values or return 'N/A' if the file name is None
    def safe_calculate_cer_values(automated_results_dir, mistral_file, results_test_trocr_dir, ocr_file):
        if mistral_file and ocr_file:
            return calculate_cer_values(automated_results_dir, mistral_file, results_test_trocr_dir, ocr_file)
        else:
            return 'N/A'

    # Function to calculate label change percentages or return 'N/A' if the file name is None
    def safe_calculate_label_change_percentages(automated_results_dir, mistral_file):
        if mistral_file:
            return calculate_label_change_percentages(automated_results_dir, mistral_file)
        else:
            return 'N/A'

    # Generate CER values for OCR and Mistral
    cer_values_ocr_self = safe_calculate_cer_values(automated_results_dir, self_training_file_name_mistral_75,
                                                    results_test_trocr_dir, self_training_file_name_ocr_75)
    cer_values_ocr_final_75 = safe_calculate_cer_values(automated_results_dir, final_test_file_name_mistral_75,
                                                        results_test_trocr_dir, final_test_file_name_ocr_75)
    cer_values_ocr_final_75_25 = safe_calculate_cer_values(automated_results_dir, final_test_file_name_mistral_75_25,
                                                           results_test_trocr_dir, final_test_file_name_ocr_75_25)

    # Generate label change percentages for Mistral
    label_changes_self = safe_calculate_label_change_percentages(automated_results_dir,
                                                                 self_training_file_name_mistral_75)
    label_changes_final_75 = safe_calculate_label_change_percentages(automated_results_dir,
                                                                     final_test_file_name_mistral_75)
    label_changes_final_75_25 = safe_calculate_label_change_percentages(automated_results_dir,
                                                                        final_test_file_name_mistral_75_25)

    # Create a dictionary to store the results
    results_dict = {
        "Self-training": cer_values_ocr_self,
        "Final test normal": cer_values_ocr_final_75,
        "Final test with Self-training": cer_values_ocr_final_75_25,
        "Labeling changes Self-training": label_changes_self,
        "Labeling changes Final test normal": label_changes_final_75,
        "Labeling changes Final test Self-training": label_changes_final_75_25
    }

    return results_dict
