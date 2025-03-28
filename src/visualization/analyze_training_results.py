import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_training_results(results, save_dir="reports/figures/training_analysis"):
    """Trains models, collects loss and accuracy, and generates visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss comparison
    plt.figure(figsize=(10, 5))
    for model_name, data in results.items():
        plt.plot(data["loss"], label=model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Models")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_comparison.png"))
    plt.close()
    
    # Accuracy comparison
    plt.figure(figsize=(10, 5))
    for model_name, data in results.items():
        plt.plot(data["accuracy"], label=model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Models")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_comparison.png"))
    plt.close()
    
    # Confusion matrices
    for model_name, data in results.items():
        cm = confusion_matrix(data["y_test"], data["predictions"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
        plt.close()
    
    # Print final accuracies
    for model_name, data in results.items():
        final_acc = data["accuracy_train"]
        print(f"{model_name}: Final Accuracy = {final_acc:.4f}")
