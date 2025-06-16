import os
import sys
import mlflow
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer
from skopt.callbacks import VerboseCallback

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    n_estimators_max = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth_max = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots_vs_users_preprocessing.pkl")

    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    X_train = loaded_data['X_train']
    X_test = loaded_data['X_test']
    y_train = loaded_data['y_train']
    y_test = loaded_data['y_test']

    #mlflow.set_experiment("Latihan MSML")

    input_example = X_train[0:1]

    search_space = {
        'n_estimators': Integer(10, n_estimators_max),
        'max_depth': Integer(1, max_depth_max)
    }
    n_iter = 50

    def mlflow_callback(res):
        params = res.x_iters[-1]
        param_names = list(opt.search_spaces.keys())
        param_dict = dict(zip(param_names, params))
        score = res.func_vals[-1]
        for k, v in param_dict.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("cv_accuracy", score)

    opt = BayesSearchCV(
        RandomForestClassifier(random_state=42),
        search_spaces=search_space,
        n_iter=n_iter,
        cv=2,
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="bayes_search_rf"): #, nested=True):
        opt.fit(
            X_train, y_train,
            callback=[mlflow_callback, VerboseCallback(n_total=n_iter)]
        )
    
        best_params = opt.best_params_
        best_score = opt.best_score_
        mlflow.log_param("n_estimators", best_params['n_estimators'])
        mlflow.log_param("max_depth", best_params['max_depth'])
        mlflow.log_metric("cv_accuracy", best_score)
        mlflow.sklearn.log_model(opt.best_estimator_, "model", input_example=input_example)