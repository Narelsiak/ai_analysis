import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
    
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X_train, X_test=None):
        self.fit(X_train)
        X_train_scaled = self.transform(X_train)
        if X_test is not None:
            X_test_scaled = self.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled

class MeanStdScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X_train, X_test=None):
        self.fit(X_train)
        X_train_scaled = self.transform(X_train)
        if X_test is not None:
            X_test_scaled = self.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled
