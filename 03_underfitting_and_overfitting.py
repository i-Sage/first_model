"""
    Models can suffer from either:

1. Overfitting: capturing supurious patterns that won't recur in the future, leading to
   less accurate predictions, or
2. Underfitting: failing to capture relevant patterns, again leading to less accurate pre   dictions.

"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def get_mae(max_leaf_nodes, train_X, val_X, train_Y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_Y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)


melbourne_data = pd.read_csv("melb_data.csv")
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price

melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Relationship between the max leaf nodes and MAE
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Max leaf nodes: {max_leaf_nodes} \t\t Mean mean_absolute_error: {my_mae}")
