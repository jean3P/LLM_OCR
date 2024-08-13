# Extracted data from the provided tables
from src.utils.visual_representation_1 import plot_cer_performance

training_percentages = ["25%", "50%", "75%", "100%"]
self_training_scenarios = ["25+75%", "50+50%", "75+25%"]

# Normal training subset data for TrOCR
trocr_ocr_cer_normal = [30.11, 25.57, 19.65, 18.81]
trocr_mistral_cer_normal = [30.03, 23.26, 17.81, 20.26]

# Normal training subset data for HTR-Flor
htrflor_ocr_cer_normal = [5.6, 5.69, 5.08, 5.16]
htrflor_mistral_cer_normal = [6.11, 6.74, 6.96, 6.96]

# Self-training subset data for TrOCR
trocr_ocr_cer_self = [26.34, 20.5, 17.62]
trocr_mistral_cer_self = [25.02, 19.22, 19.09]

# Self-training subset data for HTR-Flor
htrflor_ocr_cer_self = [5.51, 6.05, 5.55]
htrflor_mistral_cer_self = [6.42, 7.07, 6.68]

# Call the function with the data
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
