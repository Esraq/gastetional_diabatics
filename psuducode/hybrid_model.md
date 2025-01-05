
Input: Dataset with features (X) and target variable (y).
Data Preparation:
Split the dataset into training (70%) and testing (30%) subsets, ensuring class distribution is preserved.
Define Base Models:
Random Forest (RF): Handles imbalanced data and provides feature importance.
Gradient Boosting Machine (GBM): Captures nonlinear relationships.
Support Vector Machine (SVM): Finds optimal decision boundaries.
Define Stacking Ensemble:
Use RF, GBM, and SVM as base learners.
Employ Logistic Regression as the meta-learner to combine outputs from base models.
Apply cross-validation (e.g., 5-fold) to train the meta-learner on predictions from base models.
Train the Hybrid Model:
Fit the stacking ensemble on the training data.
Make Predictions:
Predict class labels for the test set.
Predict probabilities for ROC-AUC calculation.
Evaluate Model:
Compute metrics: accuracy, precision, recall, F1-score, and ROC-AUC.
Output:
Display evaluation metrics for analysis.