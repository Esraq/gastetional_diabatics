import matplotlib.pyplot as plt

# Data
models = ['Random Forest', 'SVM', 'GBM', 'Hybrid Model']
roc_auc = [0.88, 0.85, 0.86, 0.91]

# Define colors for each bar
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(models, roc_auc, color=colors)
plt.xlabel('Models')
plt.ylabel('ROC-AUC')
plt.title('Comparison of Model ROC-AUC Scores')
plt.ylim(0.8, 1.0)  # Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
