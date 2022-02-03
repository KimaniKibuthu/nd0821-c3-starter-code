# Script to train machine learning model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


# Add the necessary imports for the starter code.

# Add code to load in the data.
census = pd.read_csv('starter\data\census.csv', index_col=False)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(census, test_size=0.20, random_state=42)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Preprocess train data
X_train, y_train, encoder, lb, train_column_names = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


# Preprocess test data
X_test, y_test, encoder, lb, test_column_names = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


# Train model
model, score = train_model(X_train, y_train)
predictions = inference(model, X_test)
metrics = compute_model_metrics(y_test, predictions)


# Print
print(f'The train fbeta score is {score}. The inference performance on test data reveals a precision of {metrics[0]}, a recall of {metrics[1]} and an fbeta score of {metrics[2]}')

# Save cleaned data
cleaned_train = np.concatenate([X_train, np.reshape(y_train, (-1,1))], axis=1, )
cleaned_test = np.concatenate([X_test, np.reshape(y_test, (-1,1))], axis=1)

np.savetxt("starter\data\clean_census_train.csv", cleaned_train, delimiter=",", fmt='%.0f', header=",".join([name for name in train_column_names]))
np.savetxt("starter\data\clean_census_test.csv", cleaned_test, delimiter=",", fmt='%.0f', header=",".join([name for name in test_column_names]))
#pd.to_csv('starter\data\cleaned_census_train.csv', cleaned_train)
#pd.to_csv('starter\data\cleaned_census_test.csv', cleaned_test)