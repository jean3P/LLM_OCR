# Extracted data from the provided tables
from src.utils.visual_representation_1 import plot_cer_performance

training_percentages = ["25%", "50%", "75%", "100%"]
self_training_scenarios = ["25+75%", "50+50%", "75+25%"]

# Normal training subset data for TrOCR
trocr_ocr_cer_normal = [30.11, 25.57, 19.65, 18.81]
trocr_mistral_cer_normal = [29.74, 23.7, 19.92, 22.62]

# Normal training subset data for HTR-Flor
htrflor_ocr_cer_normal = [5.74, 6.32, 4.98, 6.39]
htrflor_mistral_cer_normal = [9.18, 10.13, 9.33, 9.45]

# Self-training subset data for TrOCR
trocr_ocr_cer_self = [27.98, 22.49, 17.52]
trocr_mistral_cer_self = [27.19, 25.74, 19.56]

# Self-training subset data for HTR-Flor
htrflor_ocr_cer_self = [7.88, 6.79, 5.85]
htrflor_mistral_cer_self = [11.56, 9.6, 9.64]

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