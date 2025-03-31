import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder  # Added for relabeling classes

def train_and_evaluate(x_train, x_test, y_train, y_test):
    """
    Train the XGBoost Classifier and evaluate its performance.
    """

    # Debugging: Print unique class labels before filtering
    print("Unique values in y_train before filtering:", np.unique(y_train))
    print("Unique values in y_test before filtering:", np.unique(y_test))

    # Define valid classes (Excluding classes 8, 9, 13)
    valid_classes = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14}

    # Filter training data
    train_mask = y_train.isin(valid_classes)
    x_train, y_train = x_train[train_mask], y_train[train_mask]

    # Filter testing data
    test_mask = y_test.isin(valid_classes)
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Debugging: Print unique class labels after filtering
    print("Unique values in y_train after filtering:", np.unique(y_train))
    print("Unique values in y_test after filtering:", np.unique(y_test))

    # Relabel class numbers to be continuous (0, 1, 2, ..., 11)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Debugging: Check class labels after remapping
    print("After relabeling, unique values in y_train:", np.unique(y_train))
    print("After relabeling, unique values in y_test:", np.unique(y_test))

    # Calculate class weights (Avoid division by zero)
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    scale_pos_weight = [total_samples / (len(class_counts) * (class_counts[i] + 1e-6)) for i in range(len(class_counts))]

    # Load the StandardScaler for feature scaling
    scaler = joblib.load('scaler_classifier.pkl')
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Get the number of unique classes dynamically
    num_classes = len(np.unique(y_train))

    # Initialize the XGBoost Classifier with improved hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=300,  # Increased estimators for better learning
        max_depth=10,  # Increased depth for better feature learning
        learning_rate=0.03,  # Lower learning rate for stability
        subsample=0.8,  # Prevents overfitting by using 80% of data per tree
        colsample_bytree=0.8,  # Prevents overfitting by using 80% of features per tree
        random_state=42,
        objective="multi:softmax",  # Multi-class classification
        num_class=num_classes,  # Updated dynamically
        scale_pos_weight=scale_pos_weight  # Handles class imbalance
    )

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.6f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("\nConfusion Matrix:")
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
    print(conf_matrix)

    # Calculate additional metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nWeighted F1 Score: {f1:.6f}")

    # If using "multi:softprob", compute ROC-AUC
    if model.objective == "multi:softprob":
        y_proba = model.predict_proba(x_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        print(f"ROC-AUC Score: {roc_auc:.6f}")

    return model, label_encoder  # Returning label encoder for future decoding if needed

def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")

if __name__ == "__main__":
    # Import preprocessing functions
    from preprocessing_classifier import load_data  # ✅ Fixed import

    # File paths
    train_file_path = 'train_preprocessed.csv'
    test_file_path = 'test_preprocessed.csv'
    model_filename = 'nids_xgboost_classifier_model.pkl'
    label_encoder_filename = 'label_encoder_classes.pkl'  # Save encoder for future use

    # Load the preprocessed training and testing data
    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)

    # Extract features and labels
    X_train, y_train = df_train.drop(columns=['Label']), df_train['Label']
    X_test, y_test = df_test.drop(columns=['Label']), df_test['Label']

    # Train and evaluate the classifier model
    model, label_encoder = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Save the trained model
    save_model(model, model_filename)

    # Save the LabelEncoder to ensure consistent mapping
    joblib.dump(label_encoder, label_encoder_filename)
    print(f"\nLabelEncoder saved to {label_encoder_filename}")
