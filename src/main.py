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

def main():

    analyze_data(X, y)
    train_MLP_base(X_train, y_train, X_test, y_test)
    train_MLP_L2(X_train, y_train, X_test, y_test)
    train_MLP_ADAM(X_train, y_train, X_test, y_test)
    train_MLP_ADAM_L2(X_train, y_train, X_test, y_test)
    #train_MLP_tf(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
