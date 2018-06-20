
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from dask_searchcv import GridSearchCV
# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time

def main():

    param_space = {"C": [1e-4, 1, 1e4],
                   "gamma": [1e-3, 1, 1e3],
                   "class_weight": [None, "balanced"],
                   "kernel":["linear", "poly", "rbf", "sigmoid"]}    
    model = SVC()
    
#     param_space = {"n_estimators":[100, 200, 300, 400, 500], 
#                                    "criterion":["gini", "entropy"], 
#                                    "max_features":["auto", "sqrt", "log2"], 
#                                    "max_depth":[2, 3, 4, 5, 6, 7, 8]}
#     model = RandomForestClassifier()
    
    digits = load_digits()
    
#     classifier = GridSearchCV(model, param_space, n_jobs=-1, cv=5)

    classifier = GridSearchCV(model, param_space, n_jobs=-1, cv=5)

    classifier.fit(digits.data, digits.target)
    
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