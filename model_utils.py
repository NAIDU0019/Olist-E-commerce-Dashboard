from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def train_and_evaluate_model(data, features, target, test_size=0.2, random_state=42):
    """
    Train and evaluate a RandomForestClassifier model.

    Parameters:
        data (pd.DataFrame): The dataset containing features and target.
        features (list): List of feature column names.
        target (str): Target column name.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the trained model, label encoder, accuracy, and classification report.
    """
    # Drop rows with missing values in selected features or target
    features = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty', 
                    'product_weight_g', 'product_volume_cm3', 'category_popularity']
    target = 'price_bucket'
    ml_data = data[features + [target]].dropna().copy()

    # Encode target variable
    label_encoder = LabelEncoder()
    ml_data[target] = label_encoder.fit_transform(ml_data[target])

    # Split data into train and test sets
    X = ml_data[features]
    y = ml_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    

    return {
        "model": model,
        "label_encoder": label_encoder,
        "accuracy": accuracy,
        "classification_report": report
    }



