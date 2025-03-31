from mlp.mlp import MLP
from mlp.layers import Dense
from mlp.optimizers import Adam

adam_optimizer = Adam(learning_rate=0.001)
model = MLP([
    Dense(neurons=10, input_size=4, activation='relu'),
    Dense(neurons=8, activation='relu'),
    Dense(neurons=3, activation='softmax')
],optimizer=adam_optimizer)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_keras_iris_dataset():
    iris = load_iris()

    X = iris.data
    y = iris.target

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_one_hot = np.eye(len(np.unique(y)))[y_encoded]

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, stratify=y_one_hot)

    return X_train, X_test, y_train, y_test, X, y_one_hot

model.summary()
X_train, X_test, y_train, y_test, _, _ = load_keras_iris_dataset()
model.fit(X_train, y_train, epochs=230)
predict, acc = model.predict(X_test, y_test)
print("Accuracy: ", acc)