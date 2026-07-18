"""
Run this ONCE in your notebook/script right after you build `X` (the final
encoded feature dataframe), alongside your existing joblib.dump() calls for
best_model.pkl / scaler.pkl / feature_columns.pkl.

It saves a "typical house" template (median of every column) so the API
can fill in the ~270 columns the user doesn't manually enter.
"""

import joblib

# X = your final one-hot encoded feature dataframe from the notebook
default_values = X.median().to_dict()

joblib.dump(default_values, "default_values.pkl")
print("Saved default_values.pkl")
