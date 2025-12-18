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
y_df = pd.concat([y_train, y_test], axis=0)

test_df = pd.read_csv("test.csv")
test_df = test_df.drop(columns=["date"])
test_df = test_df.set_index("id")
real_testx_scaled = get_scaled_test_data()

def plot_feature_importance_accuracy(x_range, ac_scores, metric):

    plt.plot(list(range(1, x_range + 1)), ac_scores)
    #plt.xticks(list(range(1, len(X_train.columns) + 1)))
    plt.title(f"{metric} Score with N Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
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

def compare_two_results(path1, path2, name1, name2):
    df1 = pd.read_csv(path1).set_index("features")
    df2 = pd.read_csv(path2).set_index("features")

    fig, ax = plt.subplots(2, figsize=(10, 12)) 
    ax[0].plot(df1.index, df1["score"], label=f"{name1}")
    ax[0].set_ylabel(f"{name1}")
    ax[0].set_xlabel("Number of features")
    ax[0].set_xticks(list(range(5, 101, 5)))
    ax[0].legend()


    ax[1].plot(df2.index, df2["score"], label=f"{name2}")
    ax[1].set_xlabel("Number of Features")
    ax[1].set_ylabel(f"{name2}")
    ax[1].set_xticks(list(range(5, 101, 5)))
    ax[1].legend()

    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    test_df_cols = test_df.columns.tolist()
    mean_cols = [c for c in test_df_cols if "mean" in c]

    """
    compare_two_results(
        path1="cv_score_balanced_accuracy.csv",
        path2="cv_score_f1_weighted.csv",
        name1="Balanced accuracy",
        name2="Weighted F1 score"
    )
    """

    cr = x_train.corr(method="pearson")

    print(cr)
