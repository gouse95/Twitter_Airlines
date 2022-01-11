# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from get_data import read_params
import argparse
import joblib
import json
# NOTE: For windows user-
# This file must be created in the root of the project 
# where Training and Prediction batch file as are present

import os
from glob import glob


data_dirs = ["train_Tweets","test_Tweets"]

for data_dir in data_dirs:
    files = glob(data_dir + r"/*.csv")
    for filePath in files:
        # print(f"dvc add {filePath}")
        os.system(f"dvc add {filePath}")

print("\n #### all files added to dvc ####")

def eval_metrics(actual, pred):
    clas_rep = classification_report(actual, pred)
    return clas_rep

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
    class_weight = config["estimators"]["RandomForestClassifier"]["params"]["class_weight"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    print(test_x)

    lr = RandomForestClassifier(n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    
    (clas_rep) = eval_metrics(test_y, predicted_qualities)
     # Report training set score
    train_score = lr.score(train_x, train_y) * 100
    print(train_score)
    # Report test set score
    test_score = lr.score(test_x, test_y) * 100
    print(test_score)

    #print("Random Forest classifier (n_estimators=%f, class_weight=%f):" % (n_estimators, class_weight))
    print("  Classification_Report: %s" % clas_rep)

#####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "Classification_Report": clas_rep,
            "test_score": test_score
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "n_estimators": n_estimators,
            "class_weight": class_weight
        }
        json.dump(params, f, indent=4)
#####################################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
