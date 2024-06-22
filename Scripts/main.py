# CoolBeansProgramming
# 6/20/24
# Train a logistic regression model on customer retention data. 


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Read in data
data_folder = 'Data'
file_path = os.path.join(data_folder, 'churn_raw.csv')
customer = pd.read_csv(file_path, index_col=0)
print(customer.head())



# Remove identifying information 
cols_to_drop = [
    'Surname',
    'CustomerId']

customer.drop(columns=cols_to_drop, inplace=True)

# Create test and training sets
X = customer.iloc[:,:-1]
y = customer.iloc[:,-1]


# Split data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)


# Verify the proportion of class labels 
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# Encode and standardize data
cols_to_encode = [
    'Geography',
    'Gender'
]

cols_to_stand = [
    'CreditScore',
    'Age',
    'Balance',
    'EstimatedSalary',
    'Tenure'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, drop='first'), cols_to_encode),
        ('num', StandardScaler(),cols_to_stand)
    ],
    remainder='passthrough'  # Keep other columns
)


# Apply transformations to test and training sets
X_train_processed = preprocessor.fit_transform(X_train) # fit transformer to training
X_test_processed = preprocessor.transform(X_test) # transform (center and scale) test data using pre-fitted preprocessor


# Train the model 
lr_model = LogisticRegression(C=100.0, solver='lbfgs',multi_class='ovr',max_iter=10000)
lr_model.fit(X_train_processed, y_train)

# Make predictions on the test data
y_pred = lr_model.predict(X_test_processed)


# Evaluate the model using accuracy_score()
score = accuracy_score(y_test,y_pred)
print('Accuracy (accuracy_score)', score)

# Evaluate the model using the model's score method
accuracy = lr_model.score(X_test_processed, y_test)
print('Accuracy (model.score):', accuracy)