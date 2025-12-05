import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import shap
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score

from data import get_data, get_scaled_test_data, scale_data
from hyperparam_optimization import objective_func, run_study, cv_score
from plotting import plot_feature_importance_rsquared, plot_rf_feature_importances

np.set_printoptions(suppress=True, precision=4)

def get_random_forest_classifier(params=None) -> RandomForestClassifier:
    if params is None:
        model = RandomForestClassifier(random_state=42)
        return model
    
    model = RandomForestClassifier(**params)
    return model

def get_feature_importances(model: RandomForestClassifier, feature_names: list):
    print("\nRanking feature importances of an RF classifier...")
    ranked_features = pd.Series(
        data=model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    return ranked_features

def find_most_important_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    features: pd.Series,
    model: RandomForestClassifier
    ):
    # Find the most important features by adding 1 feature at a time 
    print("Testing feature importance starting with 0 features...") 

    best_features = []
    r2_scores = []
    cv_scores = []  
    best_r2_score = 0

    x_df = pd.concat([x_train, x_test], axis=0)
    y_df = pd.concat([y_train, y_test], axis=0)
    #y_df = pd.concat([y_train, y_test], axis=0).apply(lambda x: "nonevent" if x == "nonevent" else "event")

    for i in range(1, len(features)+1):
        print(f"Feature count: {i}") if i % 5 == 0 else None

        features_to_fit = features.index[:i]
        model.fit(x_train[features_to_fit], y_train)

        # Calculate cross validation score (computationally heavy)
        """
        cv_error = cv_score(
            model=model,
            train_y=y_df,
            train_x=x_df[features_to_fit]
        )
        cv_scores.append(cv_error)
        """

        r2_score = model.score(x_test[features_to_fit], y_test)
        r2_scores.append(r2_score) 
        
        if r2_score > best_r2_score:
            best_r2_score = r2_score
            best_features = features_to_fit

            print(f"New best result: Features = {len(best_features)}, R-squared = {round(best_r2_score, 6)}")
    print("Feature importance testing completed\n")
    return best_features, r2_scores, cv_scores

def random_forest_optimized(model, train_x, train_y, test_x, test_y):
    # Function for testing a Random Forest classifier with
    # an optimal subset of features
    pred = model.predict(test_x)

    ac_score_train = model.score(train_x, train_y)
    ac_score = accuracy_score(pred, test_y)

    print("\nTesting metrics of an optimized RF classifier:")
    print(f"Testing accuracy score: {round(ac_score, 6)}")
    print(f"Training accuracy score: {round(ac_score_train, 6)}")


def create_kaggle_solution(
    model: RandomForestClassifier,
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_x: pd.DataFrame,
    params: dict
):
    print("Creating a Kaggle submission file...")
    id_col = test_x.index

    pred = model.predict(test_x)
    prob = model.predict_proba(test_x)

    max_probs = [max(p) for p in prob]

    result = pd.DataFrame(
        {
            "id": id_col,
            "class4": pred,
            "p": max_probs
        }
    )
    result = result.set_index("id")

    result.to_csv("kaggle_submission.csv")
    print("Saved a kaggle submission to csv file")

def randomforest_pipeline():
    # Run the full randomforest classifier pipeline
    #### 1. Fetch data and normalize it ####
    print("#### Starting random forest classifier pipeline ####")
    print("Fetcing and preprocessing data...")
    train_x, test_x, train_y, test_y = get_data()
    scaled_train_x, scaled_test_x = scale_data(train_x, test_x)
    print("Data preprocessing completed\n")
    
    #### 2. Optimize hyperparameters ####
    study = run_study(
        study_name="Random forest",
        storage="none",
        objective_func=objective_func,
        n_trials=100
    )
    
    best_parameters = study.best_trial.params
    best_accuracy_score = study.best_value

    #### 3. Define feature importance ####
    rf_model = get_random_forest_classifier(params=best_parameters)
    rf_model.fit(scaled_train_x, train_y)

    # Get an ordered list of model.feature_importances_
    feature_importances = get_feature_importances(rf_model, train_x.columns.tolist())

    # Define the optimal features for Random Forest classification
    best_features, r2_scores, cv_scores = find_most_important_features(
        x_train=scaled_train_x,
        x_test=scaled_test_x,
        y_train=train_y,
        y_test=test_y,
        features=feature_importances,
        model=rf_model
    )

    # Subsets of train, test with most important features
    best_train_x = scaled_train_x[best_features]
    best_test_x = scaled_test_x[best_features]

    ##### 4. Run optimizer RF classifier and print accuracy score ####

    # Redefine the RF classifier with best parameters and features
    rf_model = get_random_forest_classifier(params=best_parameters)
    rf_model.fit(best_train_x, train_y)

    random_forest_optimized(
        model=rf_model,
        train_x=best_train_x,
        train_y=train_y,
        test_x=best_test_x,
        test_y=test_y,
    )   

    #### 5. Create kaggle submission ####

    # Get the actual testing dataset
    real_testx_scaled = get_scaled_test_data()
    create_kaggle_solution(
        model=rf_model,
        train_x=best_train_x,
        train_y=train_y,
        test_x=real_testx_scaled[best_features],
        params=best_parameters
    )

    #### 6. Plot figures ####

if __name__ == "__main__":
    randomforest_pipeline()
