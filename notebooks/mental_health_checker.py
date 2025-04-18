# -*- coding: utf-8 -*-
"""mental_health_checker.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ldXRIhSZ7xyW5_d0HIw4qT_4CloXkdBo
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

df = pd.read_csv("mental_health_dataset.csv")

df.head(10)

X = df.drop("mental_health_status", axis=1)
y = df["mental_health_status"]

print(y)

print(X)

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)

print(y_train)

"""model - Random forest classifier"""

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

"""Evaluation

clasification report and confusion matrix
"""

y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

"""making a predictive system"""

sample_Input = [[19, 7.5, 1, 1, 0, 1, 0, 1, 5]]
prediction = model.predict(sample_Input)

print("Predicted Mental Health Status:", prediction[0])

#save model
joblib.dump(model, 'mental_health.pkl')

