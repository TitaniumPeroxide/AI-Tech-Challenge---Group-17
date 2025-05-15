import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
import seaborn as sns

# Load the dataset
try:
    data = pd.read_csv("ML for Weights.csv")
except FileNotFoundError:
    print("Error: 'ML for Weights.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Preprocessing and Feature Engineering
# Convert 'Decision' to binary (1 for Approved, 0 for Rejected)
data['Decision'] = data['Decision'].map({'Approved': 1, 'Rejected': 0})

# Separate features and target variable
X = data.drop('Decision', axis=1)
y = data['Decision']

# Define numerical and categorical features
numerical_features = ['Age', 'Duration in Ireland', 'Current Income level',
                      'Monthly Utility Bill', 'Monthly Phone Bill',
                      'Monthly Internet Bill', 'Monthly Rent (if applicable)']
categorical_features = ['Residency Status', 'Current Employment Status',
                      'Highest Education', 'Field of Education', 'Evidence of Income']

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)  # Use sparse_output=False
    ],
    remainder='passthrough'  # Keep other columns if any
)

# Create a pipeline with preprocessing and Logistic Regression
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Get feature names after one-hot encoding
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

# Get feature importance (coefficients)
try:
    feature_importance = pipeline.named_steps['classifier'].coef_[0]
except AttributeError:
    print("Warning: Feature importance is not available for this model (LogisticRegression with multi-class). Returning zero importance for all features.")
    feature_importance = np.zeros(len(feature_names))

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(feature_importance)})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# --- Sensitivity Analysis ---
def calculate_sensitivity(model, X, y, feature_name, perturbation_percentage=0.05):
    """
    Calculates the sensitivity of the model's prediction to a feature.

    Parameters:
    model: Trained sklearn model.
    X: Input features (DataFrame).
    y: Target variable (Series).
    feature_name: Name of the feature to perturb.
    perturbation_percentage: Percentage by which to perturb the feature.

    Returns:
    float: Change in model output (AUC) due to the perturbation.
    """
    X_perturbed = X.copy()
    original_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    # Find the column index of the feature
    feature_index = -1
    for i, col in enumerate(X_perturbed.columns):
        if feature_name in col:
            feature_index = i
            break
    if feature_index == -1:
        print(f"Feature '{feature_name}' not found in DataFrame.")
        return 0  # Return 0 if feature not found

    # Perturb the feature
    if X_perturbed.iloc[:, feature_index].dtype in [np.int64, np.float64]:
        perturbation_value = X_perturbed.iloc[:, feature_index] * perturbation_percentage
        X_perturbed.iloc[:, feature_index] += perturbation_value
    else:  # For categorical features, perturb by changing to a different category
        unique_values = X_perturbed.iloc[:, feature_index].unique()
        if len(unique_values) > 1:
            #change_to = unique_values[1]
            X_perturbed.iloc[:, feature_index] = np.random.choice(unique_values, size=len(X_perturbed), replace=True)

    perturbed_auc = roc_auc_score(y, model.predict_proba(X_perturbed)[:, 1])
    return perturbed_auc - original_auc

# Calculate sensitivity for each feature
sensitivity_results = {}
for feature_name in X.columns:
    sensitivity = calculate_sensitivity(pipeline, X_test, y_test, feature_name)
    sensitivity_results[feature_name] = sensitivity

# Print Feature Importance and Sensitivity
print("\n--- Feature Importance ---")
print(importance_df.to_string(index=False))

print("\n--- Feature Sensitivity (Change in AUC) ---")
for feature, sensitivity in sensitivity_results.items():
    print(f"{feature}: {sensitivity:.4f}")

# ---  Additional Output ---
# 1.  Model Performance on Test Set
print("\n--- Model Performance on Test Set ---")
print(f"AUC: {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]):.4f}")

# 2.  ROC Curve
fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# 3. Feature Importance Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.show()

# 4. Distribution of Predicted Probabilities
predicted_probabilities = pipeline.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
plt.figure(figsize=(8, 6))
sns.histplot(predicted_probabilities, bins=20, kde=True, color='skyblue')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability (Approved)')
plt.ylabel('Frequency')
plt.show()

# 5. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, predicted_probabilities)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
