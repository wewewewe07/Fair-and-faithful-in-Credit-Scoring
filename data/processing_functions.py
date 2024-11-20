"""Routines for processing data.

This code originates from one of my other projects:
https://github.com/dylan-slack/
Modeling-Uncertainty-Local-Explainability/blob/main/bayes/data_routines.py."""
import numpy as np
import pandas as pd

# The number of segments to use for the images
NSEGMENTS = 20
PARAMS = {
    'protected_class': 1,
    'unprotected_class': 0,
    'positive_outcome': 1,
    'negative_outcome': 0
}
IMAGENET_LABELS = {
    'french_bulldog': 245,
    'scuba_diver': 983,
    'corn': 987,
    'broccoli': 927
}


def get_and_preprocess_german():
    """"Handle processing of German.  We use a preprocessed version of German from Ustun et. al.
    https://arxiv.org/abs/1809.06514.  Thanks Berk!
    Parameters:
    ----------
    params : Params
    Returns:
    ----------
    Pandas data frame x_values of processed data, np.ndarray y_values, and list of column names
    """
    positive_outcome = 1
    negative_outcome = 0

    x_values = pd.read_csv("c:\\Users\\Dell V3400\\OneDrive\\Tài liệu\\machine learning\\ML1 Project\\data\\german_raw.csv")
    y_values = x_values["GoodCustomer"]
    loan_purpose = x_values["PurposeOfLoan"]

    unique_purposes = np.unique(loan_purpose.values)
    new_cols = np.zeros((len(loan_purpose), len(unique_purposes)))
    for i, purpose in enumerate(loan_purpose):
        indx = list(unique_purposes).index(purpose)
        new_cols[i, indx] = 1

    x_values = x_values.drop(["GoodCustomer", "PurposeOfLoan"], axis=1)

    for i, purpose in enumerate(unique_purposes):
        x_values["loanpurpose"+purpose] = new_cols[:, i]

    print(len(x_values.columns))

    x_values['Gender'] = [1 if v == "Male" else 0 for v in x_values['Gender'].values]

    y_values = np.array([positive_outcome if p ==
                         1 else negative_outcome for p in y_values.values])
    categorical_features = [0, 1, 2] + list(range(9, x_values.shape[1]))

    output = {
        "x_values": x_values,
        "y_values": y_values,
        "column_names": list(x_values.columns),
        "cat_indices": categorical_features,
    }

    return output


def get_dataset_by_name(name):
    """Gets a data set by name.

    Arguments:
        name: the name of the dataset.
    Returns:
        dataset: the dataset.
    """
    if name == "compas":
        dataset = get_and_preprocess_compas_data()
    elif name == "german":
        dataset = get_and_preprocess_german()
    else:
        message = f"Unkown dataset {name}"
        raise NameError(message)
    dataset['name'] = name
    return dataset
