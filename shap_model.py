import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from data import get_data

import shap

def shap_scores():
    test = pd.read_csv("test.csv")
    test = test.drop(columns=["date", "partlybad"])
    test = test.set_index("id")

    train_x, test_x, train_y, test_y = get_data()

    scaler = StandardScaler()

    train_x_sc = scaler.fit_transform(train_x)
    test_x_sc = scaler.transform(test_x)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_x_sc, train_y)

    # Create an explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(test_x_sc)

    #print(shap_values)
    print(model.classes_)

    # Plot SHAP summary plot
    shap.summary_plot(shap_values[:, :, 0], test_x_sc, feature_names=test.columns)
