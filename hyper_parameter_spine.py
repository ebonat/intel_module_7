
# Python Data Science Handbook
# https://jakevdp.github.io/PythonDataScienceHandbook/index.html

# Random Forest using GridSearchCV - VERY IMPORTANT!
# https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv

# Dask-ML
# http://dask-ml.readthedocs.io/en/latest/index.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier    
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib 
from sklearn.model_selection import cross_val_score

def main():
#     read dataset_spine.csv file
    df_diabetes = pd.read_csv(filepath_or_buffer="dataset_spine.csv")
     
#     define x features and y label (target)
    X = df_diabetes.drop(labels="class", axis=1)    
    y = df_diabetes["class"]
    
#     data split to select train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
#     standard scaler for x features
    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
#     1. MULTI-LAYER PERCEPTRON CLASSIFIER
#     hyper_parameter_candidates = [{"hidden_layer_sizes":[(5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20)], 
#                              "max_iter":[500, 1000, 1500, 2000], 
#                              "activation":["identity", "logistic", "tanh", "relu"],
#                              "solver":["lbfgs", "sgd", "adam"]}]
#     classifier = GridSearchCV(estimator=MLPClassifier(), param_grid=hyper_parameter_candidates, n_jobs=-1, cv=5)
    
#     2. SUPPORT VECTOR MACHINE CLASSIFIER
    hyper_parameter_candidates = [{"C":[1.0, 10.0, 100.0,], 
                                    "kernel":["linear", "poly", "rbf", "sigmoid"],
                                    "gamma":[1, 10, 100]}]     
    classifier = GridSearchCV(estimator=SVC(), param_grid=hyper_parameter_candidates, n_jobs=-1, cv=5)
    
#     3. RANDOM FOREST CLASSIFIER
#     hyper_parameter_candidates = [{"n_estimators":[100, 200, 300, 400, 500], 
#                                    "criterion":["gini", "entropy"], 
#                                    "max_features":["auto", "sqrt", "log2"], 
#                                    "max_depth":[2, 3, 4, 5, 6, 7, 8]}]
#     classifier = GridSearchCV(estimator=RandomForestClassifier(), param_grid=hyper_parameter_candidates, n_jobs=-1, cv=5)
         
#     fit the classifier model
    classifier.fit(X_train, y_train)    
    
    print("Grid Scores:")  
    means = classifier.cv_results_["mean_test_score"]
    standard_deviations = classifier.cv_results_["std_test_score"]
    for mean, standard_deviation, parameter in zip(means, standard_deviations, classifier.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, standard_deviation * 2, parameter))
    print()
    
    print("Best Score: %0.3f" % (classifier.best_score_))
    print()
    print("Best Parameters:") 
    print(classifier.best_params_)
    print()    
    

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = str(round(end_time - start_time, 1))
    minutes = str(round((end_time - start_time) / 60, 1))
    print("Program Runtime:")
    print("Seconds: {}".format(seconds))
    print("Minutes: {}".format(minutes))
    