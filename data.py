import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score

def get_scaled_test_data():
    # Get the actual test data, scaled
    test = pd.read_csv("test.csv")
    test = test.drop(columns=["date", "partlybad"])
    test = test.set_index("id")

    df_train = pd.read_csv("train.csv")
    df_train = df_train.drop(columns=["date", "partlybad"])
    df_train = df_train.set_index("id")

    df_train_x, df_train_y = df_train.drop(columns=["class4"]), df_train["class4"]

    scaler = StandardScaler()

    train_x_sc = scaler.fit_transform(df_train_x)
    test_x_sc = scaler.transform(test)

    test_x_sc = pd.DataFrame(test_x_sc, columns=test.columns, index=test.index)

    return test_x_sc

def get_data():
    # Get data where y is binary (0, 1)
    df_train = pd.read_csv("train.csv")
    df_train = df_train.drop(columns=["date", "partlybad"])
    df_train = df_train.set_index("id")

    # Convert to binary
    #df_train["class2"] = df_train["class4"].apply(lambda x: "nonevent" if x == "nonevent" else "II")
    #df_train_x, df_train_y = df_train.drop(columns=["class4", "class2"]), df_train["class2"]
    df_train_x, df_train_y = df_train.drop(columns=["class4"]), df_train["class4"]

    x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.2)
    return x_train, x_test, y_train, y_test

def scale_data(x_train, x_test):
    # Normalize data with StandardScaler
    scaler = StandardScaler()

    # The scaler return numpy arrays
    train_x_sc = scaler.fit_transform(x_train)
    test_x_sc = scaler.transform(x_test)

    # Convert the scaled data back to dataframes
    train_x_sc = pd.DataFrame(train_x_sc, columns=x_train.columns, index=x_train.index)
    test_x_sc = pd.DataFrame(test_x_sc, columns=x_test.columns, index=x_test.index)
    return train_x_sc, test_x_sc
