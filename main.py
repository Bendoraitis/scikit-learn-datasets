import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


'''
:Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude
    - MedHouseVal   median value in 100 000 USD 
'''

# Create a DataFrame from the dataset
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['target'] = california_housing.target

# Checking if exists NaN values
if df.isna().any().any():
    print("\nDataFrame contains NaN values.")
else:
    print("All info without NaN\n")

# Convert the regression task into a binary classification task
threshold = 2.0  # Example threshold to classify as 1 or 0
df['target'] = (df['target'] > threshold).astype(int)

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
log_clf = LogisticRegression(random_state=42)
log_clf_model = log_clf.fit(X_train_scaled, y_train)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_model = rnd_clf.fit(X_train_scaled, y_train)
svm_clf = SVC(gamma="scale", random_state=42)
svm_clf_model = svm_clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
logistic_regression_predictions = log_clf_model.predict(X_test_scaled)
random_classifier_predictions = rnd_clf_model.predict(X_test_scaled)
svc_predictions = svm_clf_model.predict(X_test_scaled)

# Evaluate the model
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
random_classifier_accuracy = accuracy_score(y_test, random_classifier_predictions)
svc_accuracy = accuracy_score(y_test, svc_predictions)

print(f"Logistic regression accuracy: {logistic_regression_accuracy:.2f}")
print(f"Random classifier accuracy: {random_classifier_accuracy:.2f}")
print(f"C-Support Vector Classification accuracy: {svc_accuracy:.2f}")
