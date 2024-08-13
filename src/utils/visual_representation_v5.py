import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Suppress the specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module='pandas')
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')

# Data from the table
data = {
    'OCR CER': [30.11, 30.11, 21.398, 21.398, 20.187, 20.187, 20.928, 20.928],
    'Check': ['First', 'Second', 'First', 'Second', 'First', 'Second', 'First', 'Second'],
    'Check CER': [23.946, 24.225, 17.344, 17.844, 20.432, 21.878, 18.369, 18.464]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the data
plt.figure(figsize=(12, 6))

# Plot Check CER
sns.lineplot(x='OCR CER', y='Check CER', hue='Check', style='Check', markers=True, data=df, palette='tab10', linewidth=2.5, markersize=10)

# Customize the plot
plt.title('Evaluating OCR Labels Through Iterative Checks')
plt.xlabel('OCR CER')
plt.ylabel('Check CER')
plt.legend(title='Check Stage', loc='upper left')
plt.grid(True)

# Display the plot
plt.show()
