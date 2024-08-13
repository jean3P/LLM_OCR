import matplotlib.pyplot as plt
import numpy as np

# Data from the table
training_scenario = "50% + 50%"
models = ["TrOCR", "HTR-Flor"]
ocr_cer = [22.81, 6.9]
mistral_cer = [19.68, 7.19]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))  # the label locations

# Create lines for OCR CER and Mistral CER
ax.plot(x, ocr_cer, marker='o', linestyle='-', color='blue', label='OCR CER')
ax.plot(x, mistral_cer, marker='s', linestyle='-', color='green', label='Mistral CER')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Model')
ax.set_ylabel('CER (%)')
ax.set_title(f'Performance Comparison - {training_scenario}')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()
plt.show()
