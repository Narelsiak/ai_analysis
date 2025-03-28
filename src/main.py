import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import X_train, y_train, X_test, y_test, X, y

from src.models.MLP_base import train as train_MLP_base
from src.models.MLP_L2 import train as train_MLP_L2
from src.models.MLP_ADAM import train as train_MLP_ADAM
from src.models.MLP_ADAM_L2 import train as train_MLP_ADAM_L2
from src.models.MLP_tf import train as train_MLP_tf

from src.visualization.explore_data import analyze_data
from src.visualization.analyze_training_results import analyze_training_results

def main():
    #analyze_data(X, y)
    results = {}

    models = {
        "MLP Base": train_MLP_base,
        "MLP L2": train_MLP_L2,
        "MLP Adam": train_MLP_ADAM,
        "MLP Adam + L2": train_MLP_ADAM_L2
    }

    for model_name, model_func in models.items():
        loss_history, accuracy_history, y_test_classes, predictions, accuracy = model_func(X_train, y_train, X_test, y_test)
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
