import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import load_dataset, load_keras_iris_dataset

from src.models.MLP_base import train as train_MLP_base
from src.models.MLP_L2 import train as train_MLP_L2
from src.models.MLP_ADAM import train as train_MLP_ADAM
from src.models.MLP_ADAM_L2 import train as train_MLP_ADAM_L2
from src.models.MLP_tf import train as train_MLP_tf
from src.models.MLP_ADAM_batch import train as train_MLP_ADAM_batch
from src.models.MLP_ADAM_batch_HE_init import train as train_MLP_ADAM_batch_HE_init
from src.models.MLP_ADAM_batch_HE_lrScheduler import train as train_MLP_ADAM_batch_HE_lrScheduler


from src.visualization.explore_data import analyze_data
from src.visualization.analyze_training_results import analyze_training_results
import numpy as np
def main():
    X_train, X_test, y_train, y_test, X, y = load_dataset()
    #analyze_data(X, y)
    results = {}

    models = {
        "MLP Base": train_MLP_base,
        "MLP L2": train_MLP_L2,
        "MLP Adam": train_MLP_ADAM,
        "MLP Adam + L2": train_MLP_ADAM_L2,
        "MLP Adam Batch": train_MLP_ADAM_batch,
        "MLP Adam Batch HE Init": train_MLP_ADAM_batch_HE_init,
        "MLP Adam Batch HE Init + LR Scheduler": train_MLP_ADAM_batch_HE_lrScheduler,
        "MLP Tensorflow": train_MLP_tf
    }

    for model_name, model_func in models.items():
        loss_history, accuracy_history, y_test_classes, predictions, accuracy = model_func(X_train, y_train, X_test, y_test, learning_rate=0.01)
        results[model_name] = {
            "loss": loss_history,
            "accuracy": accuracy_history,
            "y_test": y_test_classes,
            "predictions": predictions,
            "accuracy_train": accuracy
        }

    # train_MLP_base(X_train, y_train, X_test, y_test)
    # train_MLP_L2(X_train, y_train, X_test, y_test)
    # train_MLP_ADAM(X_train, y_train, X_test, y_test)
    # train_MLP_ADAM_L2(X_train, y_train, X_test, y_test)
    # #train_MLP_tf(X_train, y_train, X_test, y_test)

    analyze_training_results(results)
if __name__ == "__main__":
    main()
