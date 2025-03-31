import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

def load_data(file_path, chunk_size=100000):
    """
    Load the dataset in chunks to reduce memory usage.
    """
    print(f"Loading data in chunks from {file_path}...")
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    df = pd.concat(chunks, ignore_index=True)
    return df

def clean_data(df):
    """
    Clean the dataset and optimize memory usage.
    """
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().dropna()

    # Convert int and float columns to lower precision
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    print(f"Optimized DataFrame: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    return df

def feature_engineering(df):
    """
    Perform feature engineering, such as scaling numerical features
    and encoding categorical labels.
    """
    # Label encoding
    if 'Label' in df.columns:
        le = LabelEncoder()
        df['Label'] = le.fit_transform(df['Label'].astype('category'))  # Reduce memory usage
        joblib.dump(le, 'label_encoder_classifier.pkl')
        print("LabelEncoder saved to 'label_encoder_classifier.pkl'.")

    # Identify numerical columns (excluding 'Label')
    numerical_columns = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Label']

    # Handle infinite values by replacing them with NaN
    df[numerical_columns] = df[numerical_columns].replace([np.inf, -np.inf], np.nan)

    # Fill missing values with the median
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

    # Save feature columns
    joblib.dump(numerical_columns, 'feature_columns.pkl')
    print(f"Feature column names saved to 'feature_columns.pkl'. Total features: {len(numerical_columns)}.")

    print("Feature engineering completed (categorical data encoded, missing values handled).")
    return df, numerical_columns

def split_and_balance_data(df):
    """
    Split the data into train-test sets and apply SMOTE to balance classes.
    """
    if 'Label' not in df.columns:
        raise KeyError("The 'Label' column is missing from the dataset.")

    # Select numeric features
    numerical_columns = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
    df = df[numerical_columns]

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nClass Distribution in Training Set (Before SMOTE):")
    print(y_train.value_counts())

    # Apply SMOTE for class balancing
    if len(set(y_train)) > 1:  # Ensure there are multiple classes
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=min(5, min(y_train.value_counts()) - 1))
        sample_size = min(50000, len(y_train))  # Use at most 50,000 samples for SMOTE to reduce memory usage
        X_train_sampled, y_train_sampled = X_train.sample(n=sample_size, random_state=42), y_train.sample(n=sample_size, random_state=42)
        X_train, y_train = smote.fit_resample(X_train_sampled, y_train_sampled)
    else:
        print("Skipping SMOTE: Only one class present after split.")

    print(f"Data split into train and test sets: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, numerical_columns):
    """
    Apply StandardScaler to numerical features.
    """
    scaler = StandardScaler()

    # Keep only numerical columns for scaling
    numerical_columns = [col for col in numerical_columns if col in X_train.columns]
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    joblib.dump(scaler, 'scaler_classifier.pkl')
    print("StandardScaler saved to 'scaler_classifier.pkl'.")
    return X_train, X_test

if __name__ == "__main__":
    combined_file_path = 'combined_cic_ids2017_30percent.csv'

    # Step 1: Load the data
    df = load_data(combined_file_path)

    # Step 2: Clean the data
    df = clean_data(df)

    # Step 3: Feature Engineering (handle missing values & encode labels)
    df, numerical_columns = feature_engineering(df)

    # Step 4: Split and balance data
    X_train, X_test, y_train, y_test = split_and_balance_data(df)

    # Step 5: Scale numerical features
    X_train, X_test = scale_features(X_train, X_test, numerical_columns)

    # Step 6: Save preprocessed data
    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df['Label'] = y_train
    test_df['Label'] = y_test

    train_df.to_csv('train_preprocessed.csv', index=False)
    test_df.to_csv('test_preprocessed.csv', index=False)

    print("\nClass Distribution in the training set (After SMOTE):")
    print(y_train.value_counts())
    print("\nClass Distribution in the testing set:")
    print(y_test.value_counts())

    print("Preprocessing Complete. Data saved to 'train_preprocessed.csv' and 'test_preprocessed.csv'.")

