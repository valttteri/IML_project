import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import shap
import optuna
import math
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    balanced_accuracy_score,
    r2_score,
    f1_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.inspection import permutation_importance

from data import get_data, get_scaled_test_data, scale_data
from hyperparam_optimization import objective_func, run_study, cv_score
from plotting import plot_feature_importance_accuracy, plot_rf_feature_importances

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true"
)

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

def get_feature_importances_permutation(model, x_test, y_test):
    result = permutation_importance(
        model, x_test, y_test, scoring="neg_log_loss", n_repeats=10, random_state=42
    )

    importances = [max(abs(i) for i in arr) for arr in result["importances"]]
    print("importances")
    print(importances[:20])


    forest_importances = pd.Series(
        data=importances,
        index=x_test.columns.tolist()
    ).sort_values(ascending=False)

    return forest_importances


def find_most_important_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    features: pd.Series,
    model: RandomForestClassifier,
    cv_scoring: str
    ):
    # Find the most important features by adding 1 feature at a time 
    print("Testing feature importance starting with 0 features...") 

    best_features = []
    ac_scores = []
    cv_scores = []  
    best_ac_score = 0
    best_cv_score = 0

    x_df = pd.concat([x_train, x_test], axis=0)
    y_df = pd.concat([y_train, y_test], axis=0)
    #y_df = pd.concat([y_train, y_test], axis=0).apply(lambda x: "nonevent" if x == "nonevent" else "event")

    for i in range(1, len(features)+1):
        print(f"Feature count: {i}") if i % 5 == 0 else None

        features_to_fit = features.index[:i]
        model.fit(x_train[features_to_fit], y_train)

        # Calculate 10-fold cross-validation score
        
        cv_error = cv_score(
            model=model,
            train_y=y_df,
            train_x=x_df[features_to_fit],
            scoring=cv_scoring
        )
        cv_scores.append(cv_error)
        
        pred = model.predict(x_test[features_to_fit])

        # Calculate a single accuracy score
        #metric = balanced_accuracy_score(y_test, pred)
        #metric = f1_score(y_test, pred, average="weighted")
        #ac_scores.append(metric) 
        
        if cv_error > best_cv_score:
            best_cv_score = cv_error
            best_features = features_to_fit
            print(f"New best result: Features = {len(best_features)}, CV score = {round(cv_error, 6)}")

        """
        if metric > best_ac_score:
            best_ac_score = metric
            best_features = features_to_fit
            print(f"New best result: Features = {len(best_features)}, Score = {round(metric, 6)}")
        """
    print("Feature importance testing completed\n")

    # Save the cross-validation results to a csv file
    cv_df = pd.DataFrame({
        "features": list(range(1, len(cv_scores)+1)),
        "score": cv_scores
    }).set_index("features")
    cv_df.to_csv(f"cv_score_{cv_scoring}.csv")

    return best_features, ac_scores, cv_scores

def random_forest_optimized(model, train_x, train_y, test_x, test_y):
    # Function for testing a Random Forest classifier with
    # an optimal subset of features
    pred = model.predict(test_x)
    pred_train = model.predict(train_x)

    ac_score_train = balanced_accuracy_score(pred_train, train_y)
    ac_score = balanced_accuracy_score(pred, test_y)

    print("\nTesting metrics of an optimized RF classifier:")
    print(f"Testing accuracy score: {round(ac_score, 6)}")
    print(f"Training accuracy score: {round(ac_score_train, 6)}")

    return ac_score, ac_score_train


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

    # prob_col -> P(class2 = event)
    prob_col = []

    for i, pr in enumerate(pred):
        if pr == "nonevent":
            prob_col.append(1-max_probs[i])
        else:
            prob_col.append(max_probs[i]) 

    result = pd.DataFrame(
        {
            "id": id_col,
            "class4": pred,
            "p": prob_col
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
    
    # 2. Actual
    
    #study = run_study(
    #    study_name="Random forest",
    #    storage="none",
    #    objective_func=objective_func,
    #    n_trials=100
    #)
#
    #best_parameters = study.best_trial.params
    #best_accuracy_score = study.best_value
    
    # 2. Substitute
    best_parameters =  {
        'n_estimators': 430,
        'criterion': 'log_loss',
        'max_depth': 16,
        'min_samples_split': 6,
        'min_samples_leaf': 11,
        'min_weight_fraction_leaf': 0.007231336026171747,
        'max_features': 'log2',
        'max_leaf_nodes': 399,
        'min_impurity_decrease': 0.04466531247232871
    }

    #### 3. Define feature importance ####
    rf_model = get_random_forest_classifier(params=best_parameters)
    rf_model.fit(scaled_train_x, train_y)

    # Get an ordered list of model.feature_importances_
    feature_importances = get_feature_importances(rf_model, train_x.columns.tolist())
    #plot_rf_feature_importances(feature_importances)

    # Define the optimal features for Random Forest classification
    #best_features, ac_scores, cv_scores = find_most_important_features(
    #    x_train=scaled_train_x,
    #    x_test=scaled_test_x,
    #    y_train=train_y,
    #    y_test=test_y,
    #    features=feature_importances,
    #    model=rf_model,
    #    cv_scoring="f1_weighted"
    #)
    
    best_features = feature_importances.index[:47]
    #plot_feature_importance_rsquared(len(r2_scores), r2_scores) 

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

    # Get the actual Kaggle testing dataset
    kaggle_testx_scaled = get_scaled_test_data()
    create_kaggle_solution(
        model=rf_model,
        train_x=best_train_x,
        train_y=train_y,
        test_x=kaggle_testx_scaled[best_features],
        params=best_parameters
    )

    #### 6. Plot figures ####
    #plot_rf_feature_importances(feature_importances)
    #plot_feature_importance_accuracy(len(ac_scores), ac_scores)
    #plot_feature_importance_accuracy(len(cv_scores), cv_scores, "Cross-validation")

    return best_features, best_parameters

def rf_classifier_multiple_rounds(
        features: list,
        params: dict,
        rounds: int,
):
    """
    Calculate testing metrics of multiple runs with the RF classifier,
    and return the average. 

    Parameters:
    features -> A list of best features
    params -> A dictionary with optimal parameters
    rounds -> An integer denoting the number of iterations for this function
    """

    test_scores, train_scores = [], []
    average_test, average_train = 0, 0
    max_test, max_train = -1, -1
    min_test, min_train = 2, 2

    model = get_random_forest_classifier(params=params)

    for i in range(rounds):
        # Reset the data for each round
        x_train, x_test, train_y, test_y = get_data()
        scaled_xtrain, scaled_xtest = scale_data(x_train, x_test)
        best_testx = scaled_xtest[features]
        best_trainx = scaled_xtrain[features]

        model.fit(best_trainx, train_y)

        pred = model.predict(best_testx)
        pred_train = model.predict(best_trainx)

        ac_score_train = balanced_accuracy_score(pred_train, train_y)
        ac_score = balanced_accuracy_score(pred, test_y)

        train_scores.append(ac_score_train)
        test_scores.append(ac_score)

        avg_test = sum(test_scores)/len(test_scores)
        avg_train = sum(train_scores)/len(train_scores)

        print(f"\nRound {i+1}")
        print(f"Average training accuracy: {round(average_train, 4)}")
        print(f"Average test accuracy: {round(average_test, 4)}")

        # Update the statistics
        min_test = min(min_test, ac_score)
        max_test = max(max_test, ac_score)

        min_train = min(min_train, ac_score_train)
        max_train = max(max_train, ac_score_train)

        average_test = avg_test
        average_train = avg_train

    print(f"\n#### Final results ####")
    print("Training accuracy:")
    print(f"Average = {round(average_train, 4)}")
    print(f"Max = {round(max_train, 4)}")
    print(f"Min = {round(min_train, 4)}")

    print("\nTesting accuracy:")
    print(f"Average: {round(average_test, 4)}")
    print(f"Max: {round(max_test, 4)}")
    print(f"Min: {round(min_test, 4)}")

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()
    scaled_xtrain, scaled_xtest = scale_data(x_train, x_test)

    # Run the entire pipeline
    best_features, best_params = randomforest_pipeline()

    # Get an average of multiple RF classifier scores
    rf_classifier_multiple_rounds(
        features=best_features,
        params=best_params,
        rounds=150
    )

"""
Optimized random forest clf results:

Feature selection with 10-fold cross-validation, scoring with weighted F1 scoring
Optimal number of features: 47
Training accuracy: 0.6705
Testing accuracy: 0.7367

Training data, balanced ac, 100 rounds
Average: 0.6597
Max: 0.7792
Min: 0.5807

Testing data, balanced ac, 100 rounds
Average: 0.5456
Max: 0.7725
Min: 0.3345 
"""
