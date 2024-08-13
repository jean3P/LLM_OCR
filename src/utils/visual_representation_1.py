import matplotlib.pyplot as plt
import numpy as np

def plot_cer_performance(
    training_percentages,
    trocr_ocr_cer_normal,
    trocr_mistral_cer_normal,
    htrflor_ocr_cer_normal,
    htrflor_mistral_cer_normal,
    self_training_scenarios,
    trocr_ocr_cer_self,
    trocr_mistral_cer_self,
    htrflor_ocr_cer_self,
    htrflor_mistral_cer_self
):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # TrOCR plot for normal training subsets
    x_normal = np.arange(len(training_percentages))
    axs[0, 0].plot(x_normal, trocr_ocr_cer_normal, marker='o', linestyle='-', label='OCR CER')
    axs[0, 0].plot(x_normal, trocr_mistral_cer_normal, marker='s', linestyle='-', label='Mistral CER')
    axs[0, 0].set_title('TrOCR Performance - Normal Training Subsets')
    axs[0, 0].set_xlabel('Training Scenarios')
    axs[0, 0].set_ylabel('CER (%)')
    axs[0, 0].set_xticks(x_normal)
    axs[0, 0].set_xticklabels(training_percentages)
    axs[0, 0].legend()

    # HTR-Flor plot for normal training subsets
    axs[0, 1].plot(x_normal, htrflor_ocr_cer_normal, marker='o', linestyle='-', label='OCR CER')
    axs[0, 1].plot(x_normal, htrflor_mistral_cer_normal, marker='s', linestyle='-', label='Mistral CER')
    axs[0, 1].set_title('HTR-Flor Performance - Normal Training Subsets')
    axs[0, 1].set_xlabel('Training Scenarios')
    axs[0, 1].set_ylabel('CER (%)')
    axs[0, 1].set_xticks(x_normal)
    axs[0, 1].set_xticklabels(training_percentages)
    axs[0, 1].legend()

    # TrOCR plot for self-training subsets
    x_self = np.arange(len(self_training_scenarios))
    axs[1, 0].plot(x_self, trocr_ocr_cer_self, marker='o', linestyle='-', label='OCR CER')
    axs[1, 0].plot(x_self, trocr_mistral_cer_self, marker='s', linestyle='-', label='Mistral CER')
    axs[1, 0].set_title('TrOCR Performance - Self-Training Subsets')
    axs[1, 0].set_xlabel('Training Scenarios')
    axs[1, 0].set_ylabel('CER (%)')
    axs[1, 0].set_xticks(x_self)
    axs[1, 0].set_xticklabels(self_training_scenarios)
    axs[1, 0].legend()

    # HTR-Flor plot for self-training subsets
    axs[1, 1].plot(x_self, htrflor_ocr_cer_self, marker='o', linestyle='-', label='OCR CER')
    axs[1, 1].plot(x_self, htrflor_mistral_cer_self, marker='s', linestyle='-', label='Mistral CER')
    axs[1, 1].set_title('HTR-Flor Performance - Self-Training Subsets')
    axs[1, 1].set_xlabel('Training Scenarios')
    axs[1, 1].set_ylabel('CER (%)')
    axs[1, 1].set_xticks(x_self)
    axs[1, 1].set_xticklabels(self_training_scenarios)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

# # Data for the normal training subsets
# training_percentages = ["25%", "50%", "75%", "100%"]
#
# # TrOCR data for the normal training subsets
# trocr_ocr_cer_normal = [28.63, 21.4, 20.19, 20.93]
# trocr_mistral_cer_normal = [22.25, 14.52, 25.25, 18.63]
#
# # HTR-Flor data for the normal training subsets
# htrflor_ocr_cer_normal = [7.35, 7.35, 5.37, 6.06]
# htrflor_mistral_cer_normal = [4.89, 4.89, 4.37, 6.29]
#
# # Data for the self-training subsets
# self_training_scenarios = ["25+75%", "50+50%", "75+25%"]
#
# # TrOCR data for the self-training subsets
# trocr_ocr_cer_self = [29.19, 20.73, 22.01]
# trocr_mistral_cer_self = [25.91, 19.38, 19.21]
#
# # HTR-Flor data for the self-training subsets
# htrflor_ocr_cer_self = [6.29, 6.9, 6.92]
# htrflor_mistral_cer_self = [7.19, 7.19, 5.3]
#
# # Call the function with the data
# plot_cer_performance(
#     training_percentages,
#     trocr_ocr_cer_normal,
#     trocr_mistral_cer_normal,
#     htrflor_ocr_cer_normal,
#     htrflor_mistral_cer_normal,
#     self_training_scenarios,
#     trocr_ocr_cer_self,
#     trocr_mistral_cer_self,
#     htrflor_ocr_cer_self,
#     htrflor_mistral_cer_self
# )
