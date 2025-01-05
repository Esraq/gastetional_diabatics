import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Random Forest', 'SVM', 'GBM', 'Hybrid Model']
roc_auc = [0.88, 0.85, 0.86, 0.91]

# Simulated FPR and TPR values for illustration
fpr = {
    'Random Forest': np.linspace(0, 1, 100),
    'SVM': np.linspace(0, 1, 100),
    'GBM': np.linspace(0, 1, 100),
    'Hybrid Model': np.linspace(0, 1, 100)
}
tpr = {
    'Random Forest': np.sqrt(fpr['Random Forest']) * 0.88,
    'SVM': np.sqrt(fpr['SVM']) * 0.85,
    'GBM': np.sqrt(fpr['GBM']) * 0.86,
    'Hybrid Model': np.sqrt(fpr['Hybrid Model']) * 0.91
}

# Plot ROC Curves
plt.figure(figsize=(10, 8))
for model in models:
    plt.plot(fpr[model], tpr[model], label=f"{model} (AUC = {roc_auc[models.index(model)]:.2f})")

# Add diagonal line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

# Customize plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
