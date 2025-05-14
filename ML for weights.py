import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Step 1: Create or load dataset
# Example: synthetic dataset for demo purposes - to be substituted with training persona data as csv
data = {
    "age": [25, 30, 40, 23, 35, 28, 45],
    "nationality": ["India", "Ireland", "USA", "India", "Ireland", "USA", "India"],
    "visa_status": ["Work Permit", "Citizen", "Student", "Student", "Work Permit", "Citizen", "Student"],
    "duration_in_ireland": [1, 10, 3, 0.5, 6, 8, 2],
    "highest_education": ["Bachelor", "Master", "PhD", "Bachelor", "PhD", "Bachelor", "Master"],
    "eligible": [1, 1, 1, 0, 1, 1, 0]  # Target variable
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[["age", "nationality", "visa_status", "duration_in_ireland", "highest_education"]]
y = df["eligible"]

# Step 3: Define preprocessing
categorical_features = ["nationality", "visa_status", "highest_education"]
numeric_features = ["age", "duration_in_ireland"]

# One-hot encoding for nominal categorical features
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical_features),
    ("num", "passthrough", numeric_features)
])

# Step 4: Build and train model pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Extract coefficients
feature_names = model.named_steps["preprocessing"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]

print("\nLogistic Regression Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")

print(f"Intercept: {model.named_steps['classifier'].intercept_[0]:.4f}")
