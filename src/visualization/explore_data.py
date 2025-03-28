import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_data(X, y, save_dir="reports/figures/exploreDataset"):    
    os.makedirs(save_dir, exist_ok=True)

    """Performs basic data analysis and visualization."""
    # Basic statistics for X
    print("Descriptive statistics for features (X):")
    print(X.describe())
    
    # Class distribution
    print("\nClass distribution (y):")
    print(pd.DataFrame(y).sum(axis=0))

    # Pairplot to visualize relationships between features and classes
    df = X.copy()
    df['Species'] = np.argmax(y, axis=1)  # Convert one-hot to label for visualization
    pairplot = sns.pairplot(df, hue='Species', markers=["o", "s", "D"])
    pairplot.fig.suptitle('Pairplot of features by species', y=1.02)
    pairplot.savefig(os.path.join(save_dir, "pairplot.png"))
    plt.close()

    # Plot histograms for each feature
    X.hist(bins=15, figsize=(12, 8), grid=False)
    plt.suptitle('Histograms of features', y=1.02)
    plt.savefig(os.path.join(save_dir, "histograms.png"))
    plt.close()

    # Boxplot for feature distributions
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=X)
    plt.title('Boxplot of features')
    plt.savefig(os.path.join(save_dir, "boxplot.png"))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation heatmap')
    plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"))
    plt.close()
