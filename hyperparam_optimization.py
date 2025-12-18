import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import shap
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score

from data import get_data, get_scaled_test_data, scale_data

x_train, x_test, y_train, y_test = get_data()
x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
x_df = pd.concat([x_train, x_test], axis=0)
y_df = pd.concat([y_train, y_test], axis=0)

real_testx_scaled = get_scaled_test_data()

best_features = None

def cv_score(model, train_y, train_x, scoring: str):
    cv_result = cross_validate(model, train_x, train_y, cv=5, scoring=scoring, n_jobs=-1)

    return max(cv_result["test_score"])

def objective_func(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 500),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.05),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 50, 1000),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0, 0.05),
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    }
    
    model = RandomForestClassifier(**params)
    model.fit(x_train_scaled, y_train)
    
    score = cv_score(
        model=model,
        train_y=y_df,
        train_x=x_df,
        scoring="balanced_accuracy"
    )

    return score


def run_study(study_name: str, storage: str, objective_func: callable, n_trials: int) -> optuna.study.Study:
    print("Starting hyperparameter optimization with Optuna...")    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
        storage=None,
        load_if_exists=True
    )

    study.optimize(objective_func, n_trials=n_trials)
    print("\n#### Hyperparameter optimization results ####")
    print("Best cv score", study.best_value)
    print("Best params:", study.best_trial.params)

    return study

if __name__ == "__main__":
    study_name_1 = "randomforest1"

    # When grading the notebook, the study will be loaded from the submitted database
    study_1 = (
        run_study(
            study_name=study_name_1,
            storage="none",
            objective_func=objective_func,
            n_trials=100,
        )
    )
