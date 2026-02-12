import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# ---------------- Load Data ----------------
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# ---------------- Drop unwanted columns ----------------
drop_cols = ["Cabin", "PassengerId", "Ticket", "Name"]
df_train.drop(drop_cols, axis=1, inplace=True)
df_test.drop(drop_cols, axis=1, inplace=True)

# ---------------- Encode categorical variables ----------------
sex_map = {'male': 0, 'female': 1}
df_train['Sex'] = df_train['Sex'].map(sex_map)
df_test['Sex'] = df_test['Sex'].map(sex_map)

embarked_map = {'S': 0, 'C': 1, 'Q': 2}
df_train['Embarked'] = df_train['Embarked'].map(embarked_map)
df_test['Embarked'] = df_test['Embarked'].map(embarked_map)

# ---------------- Feature & Target ----------------
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]

# ---------------- Train-Test Split ----------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Imputation ----------------
num_cols = ['Age', 'Fare']
num_imputer = SimpleImputer(strategy='median')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_val[num_cols] = num_imputer.transform(X_val[num_cols])
df_test[num_cols] = num_imputer.transform(df_test[num_cols])

cat_cols = ['Embarked']
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])
df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols])

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_final = scaler.transform(df_test)

# ---------------- Model ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- Validation ----------------
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy)

# ---------------- Test Predictions ----------------
y_test_pred = model.predict(X_test_final)
# ---------------- Save predictions + accuracy ----------------
submission = pd.DataFrame({
    "PassengerId": pd.read_csv("test.csv")["PassengerId"].astype(int),  # ensure integer
    "Survived": y_test_pred.astype(int)  # ensure integers
})

# Add a new row for validation accuracy
accuracy_row = pd.DataFrame({
    "PassengerId": ["Validation_Accuracy"],
    "Survived": [str(round(accuracy, 4))]  # accuracy as string
})

# Concatenate while keeping PassengerId integer for actual passengers
submission = pd.concat([submission, accuracy_row], ignore_index=True)

submission.to_csv("submission_with_accuracy.csv", index=False)
print("Submission file with accuracy created!")