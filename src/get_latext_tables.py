from src.generate_latext_code import generate_latex_table
from src.utils.constants import automated_resuts, results_test_trocr

# Assuming 'generate_latex_table' is defined as shown previously
# and you have the calculate_cer_values and calculate_label_change_percentages functions correctly implemented.

# Specific file names and directories from your environment
percentage_train_set = 75

# OCR Files
self_training_file_name_ocr_75 = "test_evaluation_results_75.json"
final_test_file_name_ocr_75_25 = "final_test_evaluation_results_75_25.json"
final_test_file_name_ocr_75 = "final_test_evaluation_results_100.json"

# Mistral Files
self_training_file_name_mistral_75 = "test_evaluation_from_mistral_75.json"
final_test_file_name_mistral_75 = "final_test_evaluation_from_mistral_100.json"
final_test_file_name_mistral_75_25 = "final_test_evaluation_from_mistral_75_25.json"

# Directory paths
directory_path_mistral = automated_resuts
path_directory_ocr = results_test_trocr

# Generate LaTeX table code
latex_code = generate_latex_table(
    automated_results_dir=directory_path_mistral,
    results_test_trocr_dir=path_directory_ocr,
    self_training_file_name_ocr_75=self_training_file_name_ocr_75,
    final_test_file_name_ocr_75_25=final_test_file_name_ocr_75_25,
    final_test_file_name_ocr_75=final_test_file_name_ocr_75,
    self_training_file_name_mistral_75=self_training_file_name_mistral_75,
    final_test_file_name_mistral_75=final_test_file_name_mistral_75,
    final_test_file_name_mistral_75_25=final_test_file_name_mistral_75_25
)

print(latex_code)
