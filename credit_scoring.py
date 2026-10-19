# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv("cs-training.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 3. Display Dataset
# ==========================================

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 4. Check Dataset Information
# ==========================================

print("\nDataset Information:")
print(df.info())


# ==========================================
# 5. Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 6. Remove Unnecessary Column
# ==========================================

# The dataset contains an unnamed index column
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)


# ==========================================
# 7. Feature Engineering
# ==========================================

# Create a new feature representing total past-due incidents

df["TotalPastDue"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"]
    + df["NumberOfTime60-89DaysPastDueNotWorse"]
    + df["NumberOfTimes90DaysLate"]
)


# ==========================================
# 8. Select Features
# ==========================================

features = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "TotalPastDue"
]

X = df[features]

# Target variable
y = df["SeriousDlqin2yrs"]


# ==========================================
# 9. Display Target Distribution
# ==========================================

print("\nTarget Distribution:")
print(y.value_counts())


# ==========================================
# 10. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 11. Create Machine Learning Pipeline
# ==========================================

model = Pipeline([
    
    # Fill missing values with median
    ("imputer", SimpleImputer(strategy="median")),
    
    # Scale numerical features
    ("scaler", StandardScaler()),
    
    # Logistic Regression
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


# ==========================================
# 12. Train Model
# ==========================================

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 13. Make Predictions
# ==========================================

y_pred = model.predict(X_test)

# Probability of class 1
y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 14. Calculate Evaluation Metrics
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 15. Display Results
# ==========================================

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ==========================================
# 16. Classification Report
# ==========================================

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# ==========================================
# 17. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 18. Display Confusion Matrix Graph
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Creditworthy", "Not Creditworthy"],
    yticklabels=["Creditworthy", "Not Creditworthy"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Credit Scoring - Confusion Matrix")

plt.show()


# ==========================================
# 19. Predict a New Customer
# ==========================================

# Example customer information

new_customer = pd.DataFrame({
    "RevolvingUtilizationOfUnsecuredLines": [0.30],
    "age": [35],
    "DebtRatio": [0.40],
    "MonthlyIncome": [5000],
    "NumberOfOpenCreditLinesAndLoans": [8],
    "NumberOfTimes90DaysLate": [0],
    "NumberRealEstateLoansOrLines": [1],
    "NumberOfTime30-59DaysPastDueNotWorse": [0],
    "NumberOfTime60-89DaysPastDueNotWorse": [0],
    "NumberOfDependents": [2],
    "TotalPastDue": [0]
})


# Predict
prediction = model.predict(new_customer)

probability = model.predict_proba(new_customer)[0][1]


# ==========================================
# 20. Display New Customer Prediction
# ==========================================

print("\n==========================================")
print("NEW CUSTOMER PREDICTION")
print("==========================================")

print("Risk Probability:", round(probability * 100, 2), "%")

if prediction[0] == 0:
    print("Prediction: CREDITWORTHY / LOWER RISK")
else:
    print("Prediction: NOT CREDITWORTHY / HIGHER RISK")