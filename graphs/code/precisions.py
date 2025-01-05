import matplotlib.pyplot as plt

# Data
models = ['Random Forest', 'SVM', 'GBM', 'Hybrid Model']
precisions = [79.25, 82.61, 79.59, 86.05]

# Define colors for each bar
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(models, precisions, color=colors)
plt.xlabel('Models')
plt.ylabel('Precision (%)')
plt.title('Comparison of Model Precisions')
plt.ylim(75, 90)  # Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
