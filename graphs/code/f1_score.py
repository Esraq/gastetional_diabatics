import matplotlib.pyplot as plt

# Data
models = ['Random Forest', 'SVM', 'GBM', 'Hybrid Model']
f1_scores = [71.19, 68.47, 68.42, 73.18]

# Define colors for each bar
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(models, f1_scores, color=colors)
plt.xlabel('Models')
plt.ylabel('F1-Score (%)')
plt.title('Comparison of Model F1-Scores')
plt.ylim(65, 75)  # Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
