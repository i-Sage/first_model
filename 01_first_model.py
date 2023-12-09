"""
    BUILDING A MODEL
The steps to building and using a model are:

- Define: What type of model will it be ?. A decision tree ?. Some other type of model ?
  some other parameters of the model type are specified too.

- Fit: Capture patterns from provided data. This is the heart of modelling.

- Predict: Just what it sounds like

- Evaluate: Determine how accurate the model's predicition are.

"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbourne_data = pd.read_csv("melb_data.csv")
melbourne_data = melbourne_data.dropna(axis=0)
print(melbourne_data.head())

"""
    Selecting the prediction target
By convection, the prediction target is called y. So the code we need to save the house
prices in the melbourne data is:
"""
y = melbourne_data.Price

"""
    Choosing Features
The columns that are inputted into our model (and later used to make predictions) are 
called "features". In our case, those would be the columns used to determine the home
price.

We select multiple features by providing a list of column names inside brackets. Each
item in that list should be a string.

By convection, this data is called X
"""
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(melbourne_model.predict((X.head())))

""" Model Validation """
predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))
