import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    balanced_accuracy_score,
    r2_score,
    f1_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

    forest_importances = pd.Series(
        data=importances,
        index=x_test.columns.tolist()
    ).sort_values(ascending=False)

    return forest_importances

def average_most_important_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    features: pd.Series,
    model: RandomForestClassifier,
    cv_scoring: str,
    rounds: int
):
    # Define the average best feature subset
    print("\nDefining the average best feature subset...")
    number_of_features = []

    for i in range(rounds):
        print(f"Round {i+1}")
        best_features, _, _ = find_most_important_features(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            features=features,
            model=model,
            cv_scoring="f1_weighted"
        )

        number_of_features.append(len(best_features))
    avg = sum(number_of_features)/len(number_of_features)
    
    print(f"Average feature subset length: {avg}")
    return avg



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
        
        if cv_error > best_cv_score:
            best_cv_score = cv_error
            best_features = features_to_fit
            print(f"New best result: Features = {len(best_features)}, CV score = {round(cv_error, 6)}")

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
    test_x: pd.DataFrame
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
    rounds -> Number of iterations
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
        print(f"Average training accuracy: {round(avg_train, 4)}")
        print(f"Average test accuracy: {round(avg_test, 4)}")

        # Update the statistics
        min_test = min(min_test, ac_score)
        max_test = max(max_test, ac_score)

        min_train = min(min_train, ac_score_train)
        max_train = max(max_train, ac_score_train)

        average_test = avg_test
        average_train = avg_train

    print(f"\n#### Final results with {len(features)} features ####")
    print("Training accuracy:")
    print(f"Average = {round(average_train, 4)}")
    print(f"Max = {round(max_train, 4)}")
    print(f"Min = {round(min_train, 4)}")

    print("\nTesting accuracy:")
    print(f"Average: {round(average_test, 4)}")
    print(f"Max: {round(max_test, 4)}")
    print(f"Min: {round(min_test, 4)}")

    # Save results to csv
    df = pd.DataFrame({
        "runs": list(range(1, len(test_scores)+1)),
        "test_accuracy": test_scores,
        "train_accuracy": train_scores
    }).set_index("runs")
    df.to_csv(f"{rounds}_runs_rf.csv")

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
    best_features, ac_scores, cv_scores = find_most_important_features(
        x_train=scaled_train_x,
        x_test=scaled_test_x,
        y_train=train_y,
        y_test=test_y,
        features=feature_importances,
        model=rf_model,
        cv_scoring="balanced_accuracy"
    )

    best_features = feature_importances.index[:47]

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
        test_x=kaggle_testx_scaled[best_features],
    )

    return best_features, best_parameters

if __name__ == "__main__":
    # Run the entire Random Forest classifier pipeline
    best_features, best_params = randomforest_pipeline()
