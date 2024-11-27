"""Train german model."""
import sys
from os.path import dirname, abspath

import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from data.processing_functions import get_and_preprocess_german

german_data = get_and_preprocess_german()

X_values = german_data["x_values"]
y_values = german_data["y_values"]

scalar = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.20)
cols = X_train.columns
# Save data before transformations
X_train['y'] = y_train
X_test['y'] = y_test
X_train.to_csv('./data/german_train.csv')
X_test.to_csv('./data/german_test.csv')
X_train.pop("y")
X_test.pop("y")

X_train = X_train.values
X_test = X_test.values

# Setup pipeline
cat_pipeline = Pipeline([('scaler', StandardScaler()),
                        ('cat', CatBoostClassifier())])
cat_pipeline.fit(X_train, y_train)

print("Train Score:", cat_pipeline.score(X_train, y_train))
print("Score:", cat_pipeline.score(X_test, y_test))
print("Portion y==1:", np.sum(y_test == 1)
      * 1. / y_test.shape[0])

print("Column names: ", cols)
# print("Coefficients: ", lr_pipeline.named_steps["lr"].coef_)

with open("./data/german_model_grad_tree.pkl", "wb") as f:
    pkl.dump(cat_pipeline, f)

print("Saved model!")
