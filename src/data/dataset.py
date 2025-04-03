import pandas as pd
import numpy as np
import os
def load_dataset():
    """Loads the Iris dataset from a CSV file and preprocesses it."""
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

    def split_dataset(X, y, test_size=0.2, stratify=False):
        """Splits dataset into train and test sets, ensuring balanced class distribution if stratify=True."""
        if stratify:
            class_labels = np.argmax(y, axis=1)
            
            classes, class_counts = np.unique(class_labels, return_counts=True)
            train_indices, test_indices = [], []        
            
            for i, class_ in enumerate(classes):        
                class_indices = np.where(class_labels == class_)[0]
                np.random.shuffle(class_indices)

                num_test_samples = int(class_counts[i] * test_size)
                            
                test_class_indices = class_indices[:num_test_samples]
                train_class_indices = class_indices[num_test_samples:]
                
                test_indices.extend(test_class_indices)
                train_indices.extend(train_class_indices)
            
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
        
        else:        
            num_samples = len(y)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            split_idx = int(num_samples * (1 - test_size))
            train_indices, test_indices = indices[:split_idx], indices[split_idx:]
        
        # Tworzymy zbiory X_train, X_test, y_train, y_test
        X_train, X_test = X.iloc[train_indices].values, X.iloc[test_indices].values
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test

    def data_normalization_min_max(X_train, X_test):
        """Normalizes the dataset using Min-Max scaling."""
        min_train = X_train.min(axis=0)
        max_train = X_train.max(axis=0)

        X_train_normalized = (X_train - min_train) / (max_train - min_train)
        X_test_normalized = (X_test - min_train) / (max_train - min_train)

        return X_train_normalized, X_test_normalized

    def data_normalization_scaler(X_train, X_test):
        """Normalizes the dataset using StandardScaler (mean and std normalization)."""
        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0)
        
        X_train_normalized = (X_train - mean_train) / std_train
        X_test_normalized = (X_test - mean_train) / std_train
        
        return X_train_normalized, X_test_normalized

    def load_and_preprocess_data(file_path, test_size=0.2):
        """Loads and preprocesses data, returning train and test sets."""
        X, y = load_data(file_path)
        X = data_normalization_min_max(X)
        return split_dataset(X, y, test_size)

    file_path = os.path.join('data', 'raw', 'Iris.csv')
    X, y = load_data(file_path)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, stratify=True)
    X_train, X_test = data_normalization_min_max(X_train, X_test)
    return X_train, X_test, y_train, y_test, X, y

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
def load_keras_iris_dataset():
    iris = load_iris()

    print(iris.DESCR)
    X = iris.data
    y = iris.target

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_one_hot = np.eye(len(np.unique(y)))[y_encoded]

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, stratify=y_one_hot)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X, y_one_hot