import joblib

# Load the stored feature column names
feature_columns = joblib.load('models/feature_columns.pkl')

print("Features used for training:", feature_columns)
