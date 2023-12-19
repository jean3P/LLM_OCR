import os
import pandas as pd

from handle_dataset import load_from_json
from utils.constants import results_LLM_mistral


def calculate_mean_cer(data, key):
    cer_values = [item[key]['cer'] for item in data if key in item and 'cer' in item[key]]
    return sum(cer_values) / len(cer_values) if cer_values else None


# Directory containing the JSON files
directory = results_LLM_mistral

# Store the mean CER for each file
mean_cer_data = []

# Iterate through each file in the directory
length = 0
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        length = length+1
        data = load_from_json(file_path)
        mean_cer = calculate_mean_cer(data, 'MISTRAL')
        if mean_cer is not None:
            mean_cer_data.append({'File': filename, 'Mean CER': mean_cer})

# Convert the data to a DataFrame
df = pd.DataFrame(mean_cer_data)

# Sort the DataFrame by Mean CER and get the top 3 files
df_sorted = df.sort_values(by='Mean CER').head(length)

# Display the top 3 files with the lowest mean CER
print("Top 3 files with lowest Mean CER for Mistral:")
print(df_sorted)
