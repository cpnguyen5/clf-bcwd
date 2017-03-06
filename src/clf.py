import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn import metrics, grid_search, svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import silhouette_score


def getPath():
    """
    Function takes no parameters, returning the pathway of the data csv file.

    :return: directory pathway of CSV file
    """
    abspath = os.path.abspath(__file__) # absolute pathway to file
    head_path, f_name = os.path.split(abspath)
    work_dir = os.path.split(head_path)[0] # root working dir
    csvpath = os.path.join(work_dir, 'data', 'breast-cancer-wisconsin.csv')
    return csvpath


def readCSV(path):
    """
    Function returns the contents of the csv data as a pandas DataFrame.

    :param path: pathway to CSV file
    :return: pandas DataFrame
    """
    cols = ['id', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'adhesion',
             'epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

    df = pd.read_csv(path, sep=',', header=None, names = cols, na_values='?')
    df = df.dropna() # drop NaN values
    df.replace(to_replace={'class': {2:0, 4:1}}, inplace=True) # binary classes
    return df


def data_partition(data):
    """
    Function takes a pandas DataFrame, converts it into a NumPy array, and partitions the data
    into training-testing sets.

    :param data: pandas DataFrame
    :return: tuple of partitioned dataset (X_train, X_test, y_train, y_test)
    """
    data_array = data.as_matrix()
    # Partition into 70/30 Train/Test sets
    data_train, data_test = train_test_split(data_array, random_state=2, test_size=0.30)

    n_col = data.shape[1] - 1 # last index position
    # Isolate features from labels
    X_train = data_train[:, 0:n_col] #training features
    y_train = data_train[:, n_col] #training labels
    X_test = data_test[:, 0:n_col] #testing features
    y_test = data_test[:, n_col] #testing labels
    return (X_train, X_test, y_train, y_test)


def scale(X_train, X_test):
    """
    Function takes the training & testing sets of samples/features and returns the scaled versions.

    :param X_train: Training set features
    :param X_test:  Testing set features
    :return: tuple (normalized X_train, normalized X_test)
    """
    # Fit scaler
    scaler = MinMaxScaler().fit(X_train) # scaler object fitted to training set

    # Transform
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return (scaled_X_train, scaled_X_test)


def feature_select(X_train, X_test, y_train, n_feat='all'):
    """
    Function performs univariate feature selection using sklearn.feature_selection.SelectKBest and a score function.
    SelectKBest removes all but the "k" highest scoring features. The function will return a tuple of the reduced
    features and their respective scores and p-values.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :param n_feat:  Number of features to select
    :return: tuple (selected X_train, selected X_test, scores, p-values)
    """
    # Univariate Feature Selection - chi2 (score_function)
    score_func = SelectKBest(chi2, k=n_feat).fit(X_train, y_train) #k = # features
    select_X_train = score_func.transform(X_train)
    select_X_test = score_func.transform(X_test)

    # Score Function Attributes
    scores =  score_func.scores_
    pvals = score_func.pvalues_
    return (select_X_train, select_X_test, scores, pvals)


def gridsearch(X_train, X_test, y_train):
    """
    Function determines the optimal parameters of the best classifier model/estimator by performing a grid search.
    The best model will be fitted with the Training set and subsequently used to predict the classification/labels
    of the Testing set. The function returns the "best" classifier instance, classifier predictions, best parameters,
    and grid score.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :return: tuple of (best classifier instance, clf predictions, dict of best parameters, grid score)
    """
    # Parameter Grid - dictionary of parameters (map parameter names to values to be searched)
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']},
        # {'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'degree': [2], 'kernel': ['poly']}
    ]

    # Blank clf instance
    blank_clf = svm.SVC()

    # Grid Search - Hyperparameters Optimization
    clf = grid_search.GridSearchCV(blank_clf, param_grid, n_jobs=-1)  # classifier + optimal parameters
    clf = clf.fit(X_train, y_train)  # fitted classifier
    best_est = clf.best_estimator_
    clf_pred = best_est.predict(X_test)

    best_params = clf.best_params_  # best parameters identified by grid search
    score = clf.best_score_  # best grid score
    return (best_est, clf_pred, best_params, score)


def clf(X_train, X_test, y_train):
    """
    Function instantiates the classifier model with parameters, fits the model to the Training set, and applies it to
    the Testing set for classification predictions.

    :param X_train: Training set features
    :param X_test: Testing set features
    :param y_train: Training set labels
    :return: tuple of (fitted classifier instance, classifier predictions).
    """
    model = svm.SVC(kernel='linear', C=1, gamma=0.0001)
    model = model.fit(X_train, y_train) #Fit classifier to Training set
    y_pred = model.predict(X_test) # Test classifier on Testing set
    return (model, y_pred)


def sensitivity(model_pred, target):
    """
    Function calculates the sensitivity value of the classification, given the model's prediction and true labels.

    :param model_pred: model's classification prediction
    :param target: labels (y_test)
    :return: sensitivity value
    """
    y_pred = model_pred # prediction
    y_true = target # true labels
    #Confusion Matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    TN = float(cm[0,0]) #True Negative
    FP = float(cm[0,1]) #False Positive
    FN = float(cm[1,0]) #False Negative
    TP = float(cm[1,1]) #True Positive

    #sensitivity calculation
    final_sensitivity = TP/(TP + FN)
    return final_sensitivity


def specificity(model_pred, target):
    """
    Function calculates the specificity value of the classification, given the model's prediction and true labels.

    :param model_pred: model's classification prediction
    :param target: labels (y_test)
    :return: specificity value
    """
    y_pred = model_pred #prediction
    y_true = target #true labels
    #Confusion Matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    TN = float(cm[0,0]) #True Negative
    FP = float(cm[0,1]) #False Positive
    FN = float(cm[1,0]) #False Negative
    TP = float(cm[1,1]) #True Positive

    #specificity calculation
    N = FP + TN
    TNR = TN/N
    return TNR


def accuracy(model_pred, target):
    """
    Function calculates the accuracy value of the classification, given the model's prediction and true labels.

    :param model_pred: model's classification prediction
    :param target: labels (y_test)
    :return: accuracy value
    """
    accuracy = metrics.accuracy_score(target, model_pred)
    return accuracy


def f_score(model_pred, target):
    """
    Function calculates the F1-score value of the classification, given the model's prediction and true labels. F1-score
    is the weighted average of precision & recall with a range of [0, 1].

    :param model_pred: model's classification prediction
    :param target: labels (y_test)
    :return: F1-score value
    """
    y_pred = model_pred #prediction
    y_true = target #true labels
    f1 = metrics.f1_score(y_true, y_pred)
    return f1


def precision(model_pred, target):
    """
    Function calculates the precision value of the classification, given the model's prediction and true labels.

    :param model_pred: model's classification prediction
    :param target: labels (y_test)
    :return: precision value
    """
    y_pred = model_pred #prediction
    y_true = target #true labels
    precision_score = metrics.precision_score(y_true, y_pred)
    return precision_score


def recall(model_pred, target):
    """
    Function calculates the recall value of the classification, given the model's prediction and true labels.


    :param model_pred: model's classification prediction
    :param target: labels (y_test)
    :return: recall value
    """
    y_pred = model_pred #prediction
    y_true = target #true labels
    recall_score = metrics.recall_score(y_true, y_pred)
    return recall_score


def df_auc(model, X_test, target):
    """
    Function calculates the area under the (ROC) curve based on the decision function (y_score).

    :param model: fitted classifier model
    :param X_test: Testing set features
    :param target: labels (y_test)
    :return: AUC score
    """
    y_true = target
    y_score = model.decision_function(X_test) #Predict confidence scores
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score) #calculate FPR & TPR
    auc_score = metrics.auc(fpr, tpr) #calculate AUC
    return auc_score


def plot_roc(model, X_test, target, n_features):
    """
    Function uses matplotlib to plot the ROC curve of the classifier.

    :param model: fitted classification model
    :param X_test: Testing set features (X_test)
    :param target: labels (y_test)
    :param n_features: int indicating number of features of data set
    :return: Plot of ROC curve
    """
    y_true = target
    y_score = model.decision_function(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score) #calculate FPR & TPR
    auc_score = metrics.auc(fpr, tpr) #calculate area under the curve

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic: \n (n_features = %d)' % (n_features))
    plt.show()
    plt.close()
    return


#main
if __name__ == '__main__':
    np.random.seed(2)

    # Obtain Data
    path = getPath()
    data = readCSV(path)

    # Data Partition
    X_train, X_test, y_train, y_test = data_partition(data)

    # Data Normalization
    scaled_X_train, scaled_X_test = scale(X_train, X_test)

    # Feature Selection
    n_features = 8
    select_X_train, select_X_test, score, pval = feature_select(scaled_X_train, scaled_X_test, y_train, n_feat=n_features)
    feat_names = list(data.columns)[:-1]
    # for i in range(len(feat_names)):
    #     print feat_names[i], "\n \t", "score: ", score[i], "\n \t", "p-value: ", pval[i]
    # print

    # Classification
    clf_model, y_pred, best_p, best_score = gridsearch(select_X_train, select_X_test, y_train)
    print "Best Parameters: ", best_p
    print "Best Grid Search Score: ", best_score
    print "Best Estimator: ", clf_model, "\n"
    # clf_model, y_pred = clf(select_X_train, select_X_test, y_train) #alternative clf function [no grid search]

    print "Classifier Model (SVC) Metrics: "
    print "------------------------------------------------------------------------"
    print "Accuracy: ", np.around(accuracy(y_pred, y_test), 5)
    print "Senstivity: ", np.around(sensitivity(y_pred, y_test), 5)
    print "Specificity: ", np.around(specificity(y_pred, y_test), 5)
    print "F1 Score: ", np.around(f_score(y_pred, y_test), 5)
    print "Precision: ", np.around(precision(y_pred, y_test), 5)
    print "Recall: ", np.around(recall(y_pred, y_test), 5)
    print "AUC: ", np.around(df_auc(clf_model, select_X_test, y_test), 5), "\n"

    #ROC Plot
    plot_roc(clf_model,select_X_test, y_test, n_features)