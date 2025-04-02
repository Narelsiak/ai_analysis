from mlp.mlp import MLP
from mlp.layers import Dense, BatchNormalization
from mlp.optimizers import Adam
from mlp.lr_schedules import ExponentialDecay, StepDecay

lr_schedule = ExponentialDecay(initial_lr=0.01, decay_rate=0.01)
adam_optimizer = Adam(schedule=lr_schedule)
model = MLP([
    Dense(neurons=10, input_size=4, activation='relu'),
    Dense(neurons=8, activation='relu'),
    Dense(neurons=3, activation='softmax')
], optimizer=adam_optimizer)

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
history = model.fit(X_train, y_train, epochs=230, validation=(X_test, y_test), log_level=1)
predict, acc = model.predict(X_test, y_test)
print("Accuracy: ", acc) 

import matplotlib.pyplot as plt

def plot_history(history, save_path="training_history.png"):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.savefig(save_path)  # Zapisz wykres jako plik
    print(f"Plot saved as {save_path}")

plot_history(history)


#TODO -> sprawdzenie tego val_lossa wzgledem lossa, jest mniejszy. Sprawdzenie jak wygladaja te roznice ok?
### patrzac niby po tf, to tak ma byc? Ale no, trzeba sprawdzic. W teorii loss -> dazymy do najmniejszego, val_loss -> moze poskakac, praktyka odwrotnie.