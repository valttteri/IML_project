import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import shap
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data import get_data, scale_data, get_scaled_test_data

x_train, x_test, y_train, y_test = get_data()
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
x_df = pd.concat([x_train, x_test], axis=0)
#y_df = pd.concat([y_train, y_test], axis=0).apply(lambda x: "nonevent" if x == "nonevent" else "event")
y_df = pd.concat([y_train, y_test], axis=0)

test_df = pd.read_csv("test.csv")
test_df = test_df.drop(columns=["date"])
test_df = test_df.set_index("id")
real_testx_scaled = get_scaled_test_data()

def plot_feature_importance_rsquared(x_range, r2_scores):

    plt.plot(list(range(1, x_range + 1)), r2_scores)
    #plt.xticks(list(range(1, len(X_train.columns) + 1)))
    plt.title(f"$R^2$ score with the n most important features")
    plt.xlabel("Number of features")
    plt.ylabel(f"$R^2$")
    plt.show()

def plot_rf_feature_importances(data: pd.Series):
    # Plot a bar chart displaying feature importance
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.bar(
        x=list(data.index),
        height=list(data.values)
    )

    plt.title("Feature importances using MDI")
    plt.ylabel("Mean decrease in impurity")
    plt.xticks(fontsize=7, rotation=45, ha="right")
    
    plt.show()

def pairplot_data(data):
    sns.set_theme(style="ticks")
    sns.pairplot(data=data, height=2, aspect=1.1)
    plt.show()


if __name__ == "__main__":
    test_df_cols = test_df.columns.tolist()
    mean_cols = [c for c in test_df_cols if "mean" in c]

    #print(mean_cols)

    pairplot_data(test_df[mean_cols[:5]])
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(test_df)

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=range(len(test_df)), cmap='viridis')
    plt.title('t-SNE Visualization')
    plt.colorbar(scatter, label='Classes')
    plt.show()

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(test_df)

    # Plot PCA results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(test_df)), cmap='viridis')
    plt.title('PCA Visualization of Iris Dataset')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Species')
    plt.show()

    # Print explained variance ratio
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    """
