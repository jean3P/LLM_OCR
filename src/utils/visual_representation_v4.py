# Extracted data for normal training subsets
from src.utils.visual_representation_1 import plot_cer_performance

training_percentages = ["25%", "50%", "75%", "100%"]

trocr_ocr_cer_normal = [30.11, 25.57, 19.65, 18.81]
trocr_mistral_cer_normal = [27.86, 22.41, 19.5, 18.83]

htrflor_ocr_cer_normal = [6.89, 6.55, 5.45, 5.51]
htrflor_mistral_cer_normal = [8.12, 8.43, 6.66, 6.69]

# Extracted data for self-training subsets
self_training_scenarios = ["25+75%", "50+50%", "75+25%"]

trocr_ocr_cer_self = [28.18, 17.82, 17.06]
trocr_mistral_cer_self = [25.7, 18.34, 17.76]

htrflor_ocr_cer_self = [6.59, 5.88, 4.85]
htrflor_mistral_cer_self = [6.98, 7.42, 7.49]

# Call the function with the extracted data
plot_cer_performance(
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
)
