import matplotlib.pyplot as plt

# Data
models = ['Random Forest', 'SVM', 'GBM', 'Hybrid Model']
recalls = [64.62, 58.46, 60.00, 65.92]

# Define colors for each bar
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(models, recalls, color=colors)
plt.xlabel('Models')
plt.ylabel('Recall (%)')
plt.title('Comparison of Model Recalls')
plt.ylim(50, 70)  # Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
