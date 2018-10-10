""" Trying out training a data model """
import pickle

import pandas as pd
import numpy as np
from sklearn import linear_model

# load data as a panda
data_frame = pd.read_csv('winequality-red.csv', delimiter=",")

# Get the classifier - the Quality attribute
label = data_frame['quality']

# Get all the other columns except quality
features = data_frame.drop('quality', axis=1)

# Define the linear regression estimator and train it with the wine data
regression = linear_model.LinearRegression()
regression.fit(features, label)

# Save our model using pickle
pickle.dump(regression, open("model.pkl", "wb"))

# Using our trained model to predict a fake wine
model = pickle.load(open("model.pkl", "rb"))

# Each number represents a feature
feature_array = [[11.2, 0.28, 0.56, 1.9, 0.075, 17, 60, 0.998, 3.16, 0.58, 9.8]]
prediction = model.predict(feature_array).tolist()

print (prediction)
