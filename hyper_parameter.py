
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import time
import traceback
import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier    
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib 
from sklearn.model_selection import cross_val_score

def print_exception_message(message_orientation="horizontal"):
    """
    print full exception message
   :param message_orientation: horizontal or vertical
   :return None   
    """
    try:
        exc_type, exc_value, exc_tb = sys.exc_info()           
        file_name, line_number, procedure_name, line_code = traceback.extract_tb(exc_tb)[-1]      
        time_stamp = " [Time Stamp]: " + str(time.strftime("%Y-%m-%d %I:%M:%S %p")) 
        file_name = " [File Name]: " + str(file_name) 
        procedure_name = " [Procedure Name]: " +  str(procedure_name)
        error_message = " [Error Message]: " + str(exc_value)       
        error_type = " [Error Type]: " + str(exc_type)                   
        line_number = " [Line Number]: " + str(line_number)               
        line_code = " [Line Code]: " + str(line_code)
        if (message_orientation == "horizontal"):
            print( "An error occurred:{};{};{};{};{};{};{}".format(time_stamp, file_name, procedure_name, error_message, error_type, line_number, line_code))
        elif (message_orientation == "vertical"):
            print( "An error occurred:\n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(time_stamp, file_name, procedure_name, error_message, error_type, line_number, line_code))
        else:
            pass                   
    except:
        exception_message = sys.exc_info()[0]
        print("An error occurred. {}".format(exception_message))
        
def tune_hyperparameter_model(ml_model, X_train, y_train, hyper_parameter_candidates, scoring_parameter, cv_fold, search_cv_type="grid"):   
    """
    apply grid search cv and randomized search cv algorithms to 
    find optimal hyperparameters model 
    :param ml_model: defined machine learning model
    :param X_train: feature training data
    :param y_train: target (label) training data
    :param hyper_parameter_candidates: dictionary of 
     hyperparameter candidates
    :param scoring_parameter: parameter that controls what metric 
     to apply to the evaluated model
    :param cv_fold: number of cv divided folds
    :param search_cv_type: type of search cv (gridsearchcv or 
     randomizedsearchcv)
    :return classifier_model: defined classifier model
    """
    try:
        if (search_cv_type==config.GRID_SEARCH_CV):
            classifier_model = GridSearchCV(estimator=ml_model, 
               param_grid=hyper_parameter_candidates, scoring=scoring_parameter, cv=cv_fold)
        elif (search_cv_type==config.RANDOMIZED_SEARCH_CV):
            classifier_model = RandomizedSearchCV(estimator=ml_model, param_distributions=hyper_parameter_candidates, 
               scoring=scoring_parameter, cv=cv_fold)
        classifier_model.fit(X_train, y_train)
    except:
        print_exception_message()
    return classifier_model
def main():
#     read diabetes.csv file
    df_diabetes = pd.read_csv(filepath_or_buffer="wine_data.csv")

#     define X features and y label (target)
    X = df_diabetes.drop(labels="cultivator", axis=1)    
    y = df_diabetes["cultivator"]
    
#     data split to select train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
#     standard scaler for x features
    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
#     MLP CLASSIFIER
# #     hyperparameter candidates dictonary
#     hyper_parameter_candidates = [{"hidden_layer_sizes":[(5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20)], 
#                              "max_iter":[500, 1000, 1500, 2000], 
#                              "activation":["identity", "logistic", "tanh", "relu"],
#                              "solver":["lbfgs", "sgd", "adam"]}]
# #     initialize gridsearchcv object
#     classifier = GridSearchCV(estimator=MLPClassifier(), param_grid=hyper_parameter_candidates, n_jobs=-1, cv=5)
    
#     SVC CLASSIFIER
    ml_model = SVC()
    hyper_parameter_candidates = [{"C":[1.0, 10.0, 100.0,], 
                                    "kernel":["linear", "poly", "rbf", "sigmoid"],
                                    "gamma":[1, 10, 100]}]     
    scoring_parameter = "accuracy"
    cv_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    classifier= tune_hyperparameter_model(ml_model, X_train, y_train, hyper_parameter_candidates, scoring_parameter, cv_fold)
         
    classifier.fit(X_train, y_train)    
     
    print("Grid Scores:")  
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    
    print("Best Score:", classifier.best_score_) 
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
    print("Seconds: {} | Minutes: {}".format(seconds, minutes))


    