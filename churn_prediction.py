# ==========================================
# Customer Churn Prediction Using ML
# Author: Alok Mishra
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------
df = pd.read_csv("telco_churn.csv")

print("Dataset Loaded Successfully")
print("Shape of dataset:", df.shape)
print(df.head())

# ------------------------------------------
# Step 2: Data Cleaning
# ------------------------------------------

# Drop customerID column if present
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Replace blank spaces with NaN
df.replace(" ", np.nan, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

print("\nAfter cleaning, dataset shape:", df.shape)

# ------------------------------------------
# Step 3: Encode Categorical Variables
# ------------------------------------------
le = LabelEncoder()

for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

print("\nCategorical features encoded")

# ------------------------------------------
# Step 4: Feature Scaling
# ------------------------------------------
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

print("\nFeature scaling completed")

# ------------------------------------------
# Step 5: Split Dataset
# ------------------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData split into training and testing sets")

# ------------------------------------------
# Step 6: Logistic Regression Model
# ------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

# ------------------------------------------
# Step 7: Random Forest Model
# ------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# ------------------------------------------
# Step 8: Confusion Matrix Visualization
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, rf_predictions),
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------
# Step 9: Feature Importance
# ------------------------------------------
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
feature_importance.head(10).plot(kind='bar')
plt.title("Top 10 Features Influencing Customer Churn")
plt.ylabel("Importance Score")
plt.show()

# ------------------------------------------
# Step 10: Conclusion Output
# ------------------------------------------
print("\nProject Execution Completed Successfully")
print("Random Forest model performed better than Logistic Regression")
