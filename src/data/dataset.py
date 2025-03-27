import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Loads data from a CSV file and returns X (features) and y (one-hot encoded labels)."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error while loading the file: {e}")
    
    X = df.drop(columns=['Id', 'Species'])
    y = one_hot_encoder(df['Species'])
    return X, y

def one_hot_encoder(labels):
    """Converts labels to one-hot encoding."""
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    label_indices = np.array([label_to_index[label] for label in labels])
    return np.eye(len(unique_labels))[label_indices]

def split_dataset(X, y, test_size=0.2):
    """Splits dataset into train and test sets, ensuring balanced class distribution."""
    num_samples = len(y)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X.iloc[train_indices].values, X.iloc[test_indices].values
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def data_normalization(X):
    """Normalizes the dataset using Min-Max scaling."""
    return (X - X.min()) / (X.max() - X.min())

def load_and_preprocess_data(file_path, test_size=0.2):
    """Loads and preprocesses data, returning train and test sets."""
    X, y = load_data(file_path)
    X = data_normalization(X)
    return split_dataset(X, y, test_size)

file_path = os.path.join('data', 'raw', 'Iris.csv')
X, y = load_data(file_path)
X_normalize = data_normalization(X)

X_train, X_test, y_train, y_test = split_dataset(X_normalize, y, test_size=0.2)