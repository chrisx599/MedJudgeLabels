import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Load the evaluation results file
with open('medquad_evaluation_results.json', 'r') as f:
    results = json.load(f)

# 2. Extract the confusion matrix data for 'harm_extent'
cm_data_list = results['harm_extent']['confusion_matrix']
cm_data = np.array(cm_data_list)

# 3. Define the labels (consistent with the JSON structure)
labels = ["No harm", "Moderate or mild harm", "Death or severe harm"]

# 4. Plot the Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))

# Create ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=labels)

# Plotting the display object
disp.plot(cmap="Reds", ax=ax, colorbar=False, values_format='d')

ax.set_title("Confusion Matrix – Extent of Harm (MedQuad GT vs. LLM Pred)")
ax.set_xlabel("LLM Predicted Extent of Harm")
ax.set_ylabel("MedQuad Ground Truth Extent of Harm (Support)")

# Rotate x-axis labels for better readability
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# Save the figure (optional, but good practice in an environment like this)
# fig.savefig("dynamic_medquad_vs_llm_extent_of_harm_cm.png")
plt.show() # Use plt.show() if running locally to display the plot
