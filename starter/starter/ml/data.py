from tkinter.font import names
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.
    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.
    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.
    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    updated_columns : list
        List containing the order of the columns
    """
    # Clean the column names
    column_descriptions = {'age': 'continuous.',
    'workclass': 'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.',
    'fnlwgt': 'continuous.',
    'education': 'Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.',
    'education_num': 'continuous.',
    'marital_status': 'Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.',
    'occupation': 'Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.',
    'relationship': 'Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.',
    'race': 'White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.',
    'sex': 'Female, Male.',
    'capital_gain': 'continuous.',
    'capital_loss': 'continuous.',
    'hours_per_week': 'continuous.',
    'native_country': 'Country',
    'salary':'Persons salary'
    }

    column_names = list(column_descriptions.keys())

    X.columns = column_names

    # Split data
    if label is not None:
        y = X[[label]]
        X = X.drop(label, axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features]
    X_continuous = X.drop(*[categorical_features], axis=1)

    updated_columns = np.append(X_continuous.columns, X_categorical.columns)

    if training is True:
        encoder = LabelEncoder()
        lb = LabelBinarizer()
        X_categorical = X_categorical.apply(encoder.fit_transform)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = X_categorical.apply(encoder.transform)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    
    return X, y, encoder, lb, updated_columns
