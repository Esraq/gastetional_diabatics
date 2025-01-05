import matplotlib.pyplot as plt

# Data
models = ['Random Forest', 'SVM', 'GBM', 'Hybrid Model']
accuracies = [88.82, 88.49, 88.16, 89.02]

# Define colors for each bar
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

# Create the bar chart with custom colors
plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=colors)
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Model Accuracies')
plt.ylim(85, 90)  # Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
