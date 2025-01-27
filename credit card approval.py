import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb

# Step 1: Load or create the dataset
csv_file = "credit_card_approval.csv"

if not os.path.exists(csv_file):
    data = {
        "Income": [50000, 60000, 75000, 30000, 90000, 100000, 110000, 85000, 95000, 55000],
        "Employment_Status": ["Employed", "Self-Employed", "Unemployed", "Employed", "Self-Employed", "Unemployed", "Employed", "Self-Employed", "Employed", "Unemployed"],
        "Credit_History": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "Demographics": ["Young", "Middle-Aged", "Senior", "Young", "Middle-Aged", "Senior", "Middle-Aged", "Young", "Senior", "Middle-Aged"],
        "Age": [25, 40, 60, 22, 45, 65, 35, 28, 55, 33],
        "Debt_Income_Ratio": [0.3, 0.5, 0.2, 0.7, 0.4, 0.1, 0.6, 0.3, 0.2, 0.8],  # New Feature
        "Previous_Applications": [1, 2, 0, 1, 3, 0, 2, 1, 1, 4],
        "Approval_Status": [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"✅ Dataset created: {csv_file}")
else:
    df = pd.read_csv(csv_file, encoding='latin1')
    print("✅ Dataset Loaded Successfully!\n", df.head())

# Step 2: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df["Income"], bins=10, kde=True, color="red")
plt.title("Income Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Approval_Status"], y=df["Income"])
plt.title("Income vs. Approval Status")
plt.show()

# Step 3: Preprocessing
categorical_columns = ["Employment_Status", "Demographics"]

# Apply OneHotEncoding
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features and target
X = df.drop(columns=["Approval_Status"])
y = df["Approval_Status"]

# Feature Selection: Identify most important features
feature_selector = RandomForestClassifier(n_estimators=100)
feature_selector.fit(X, y)
feature_importances = pd.Series(feature_selector.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(5).index  # Selecting top 5 features
X = X[top_features]

# Handle imbalanced data with RandomOverSampling
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss")
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\n🏆 Best Model:", best_model)

# Step 5: Fairness Analysis (Checking Model Bias)
df["Predicted"] = best_model.predict(X)
df["Demographics"] = pd.read_csv(csv_file)["Demographics"] 
sns.barplot(x=df["Demographics"], y=df["Predicted"], ci=None)
plt.title("Fairness Analysis: Predictions by Demographics")
plt.show()

# Step 6: User Input for Prediction
def get_user_input():
    print("\nPlease provide the following details:")

    try:
        income = float(input("Income (e.g., 50000): "))
        employment_status = input("Employment Status (Employed, Self-Employed, Unemployed): ").strip()
        credit_history = int(input("Credit History (1 for Good, 0 for Bad): "))
        demographics = input("Demographics (Young, Middle-Aged, Senior): ").strip()
        debt_income_ratio = float(input("Debt-to-Income Ratio (e.g., 0.3 for 30%): "))
        previous_apps = int(input("Previous Applications: "))

        return income, employment_status, credit_history, demographics, debt_income_ratio, previous_apps
    except ValueError:
        print("❌ Invalid input! Please enter valid numeric values where required.")
        return None

def predict_approval():
    """Predicts whether the user will be approved for a credit card."""
    user_input = get_user_input()
    if not user_input:
        return

    income, employment_status, credit_history, demographics, debt_income_ratio, previous_apps = user_input

    input_data = pd.DataFrame({
        "Income": [income],
        "Credit_History": [credit_history],
        "Debt_Income_Ratio": [debt_income_ratio],
        "Previous_Applications": [previous_apps]
    })

    encoded_input = pd.get_dummies(pd.DataFrame({
        "Employment_Status": [employment_status],
        "Demographics": [demographics]
    }))

    # Add missing columns with 0s
    missing_cols = [col for col in X.columns if col not in encoded_input.columns]
    for col in missing_cols:
        encoded_input[col] = 0

    # Ensure column order matches the training data
    input_data = pd.concat([input_data, encoded_input], axis=1)
    input_data = input_data.loc[:, ~input_data.columns.duplicated()]  # Remove duplicate columns
    input_data = input_data.reindex(columns=X.columns, fill_value=0)  # Ensure columns match the training data
    input_data = scaler.transform(input_data)  # Apply the same scaling as the training data

    prediction = best_model.predict(input_data)[0]
    print("\n✅ Credit Card Approval Prediction:", "Approved ✅" if prediction == 1 else "Rejected ❌")

# Call the function for prediction
predict_approval()