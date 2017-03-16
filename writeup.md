# Results

#### Classification Labels: 
The classification problem to address was binary. The classifier model would predict whether the patient/observation would
be diagnosed with benign (`0`) or malignant (`1`) breast cancer.  

Diagnosis | Class 
:--: | :--:   
Benign | 0
Malignant | 1

## Feature Selection
The goal of feature selection was to omit noisy features, which were those that provided no significance 
in the model's classification of data points. Ultimately, boosting the classifier/estimator's predictive performance. 

### Univariate Feature Selection: `SelectKBest` + `chi`<sup>2</sup> Score Function 
##### Feature Rankings (`k=5`)
Feature | Score | p-value | Ranking |
:-- | :--: | :--: | :--: |
Sample Code Number | 0.425  | 5.14e-01 | 10 |
Clump Thickness  | 64.163 | 1.146e-15 | 7 | 
**Size Uniformity**  | 163.531 | 1.915e-37 | 2 |
**Shape Uniformity** | 157.245 | 4.525e-36 | 3 |
**Marginal Adhesion** | 123.997 | 8.438e-29 | 5 |
Epithelial Cell Size | 59.817 | 1.04e-14 | 8 |
**Bare Nuclei** | 203.968 | 2.844e-46 | 1 |
Bland Chromatin | 77.769 | 1.158e-18 | 6 |
**Normal Nucleoli** | 146.994 | 7.872e-34 | 4 |
Mitoses | 52.261 | 4.860e-13 | 9 |

The top five features selected were **Bare Nuclei**, **Size Uniformity**, **Shape Uniformity**, **Normal Nucleoli**, and
**Marginal Adhesion**. Five features were selected for two primary reasons: (1) they provided the optimal performance by 
all models across the board (as indicated by the metrics) and (2) there was a large discrepancy in score function 
(*Chi<sup>2</sup>*) values between the top five features and the remainder. 

The excluded features were excluded as they were determined to be uninformative, providing little value to the model in 
its prediction of class. On a similar note, this aligns with the initial hypothesis that the **Sample Code Number** 
feature would be omitted due to the insignificance of an identification feature to the predictive classification of 
breast cancer diagnosis.

### Random Forest
##### Feature Importance
Feature | Importance | Ranking |
:-- | :--: | :--: |
Sample Code Number | 0.015 | 9 |
Clump Thickness  | 0.037 | 7 | 
Size Uniformity  | 0.223 | 1 |
Shape Uniformity | 0.127 | 4 |
Marginal Adhesion | 0.036 | 8 |
Epithelial Cell Size | 0.090 | 6 |
Bare Nuclei | 0.205 | 2 |
Bland Chromatin | 0.102 | 5 |
Normal Nucleoli | 0.160 | 3 |
Mitoses | 52.261 | 0.006 | 10 |

The Random Forest estimator implicitly performs feature selection during the time of its fitting on the training data. 
The features, in order of highest to lowest importance, were found to be: Size Uniformity, Bare Nuclei, Normal Nucleoli, 
Shape Uniformity, Bland Chromatin, Epithelial Cell Size, Clump Thickhness, Adhesion, Sample Order Number, and Mitoses.
It should also be noted that the gridsearch (hyper-optimiziation of model parameters) found that the `max_features` parameter
to be three. Thus, indicating that three features should considered during the model's splitting of decision trees. Those
three features were the top three: **Size Uniformity, Bare Nuclei, and Normal Nucleoli**.

## Classification
##### Note: Parameters Tuning
The hyper-parameters, or parameters that provide the model with its best performance, were determined using a cross-validated
grid search. It exhaustively considers all possible parameter combinations as specified in the parameter grid, implementing 
the model fitted to the training set and retaining the best combination. It should be noted that the cross-validation process
is only performed with the training data set in order to mitigate overfitting and sustain generalization of the model to 
other data sets.

### 1. Support Vector Machine (SVM)
The Support Vector Machine classifier uses a hyperplane that serves as a decision boundary to split data points and 
maximize the margin between support vectors (points closest to hyperplane). The model has a kernel function which 
defines the similarity between points and rearranges the feature space in order to make non-linear relationships linear. 
Thus, simplifying the data by increasing dimensionality. 

#### Pros
Support Vector Classifiers perform very well in high-dimensional spaces and as a model, it’s generalizable to be 
effectively applied to future datasets. In addition, it’s a versatile classifier due to the various kernels (e.g. 
linear, rbf, and poly) that enables it to handle both linear and non-linear datasets. It’s linear kernel performs almost 
equivalently to that of Logistic Regression. 

#### Cons
On the contrary, one caveat of Support Vector Classifiers is that its non-linear classifiers (e.g. rbf and poly kernels) 
have a very long run-time. For example, an increase in degree of the poly kernel increases the run time exponentially. 
Therefore, Support Vector Machine classifiers may be inefficient for industry-scale applications due to its inefficient 
and long training of the model. 

#### Parameters
kernel | C | gamma | degree 
:--: | :--: | :--: | :--: 
linear | 10 | 0.0001 | N/A 

**Cross-Validated Grid Search Score**: 0.977

A linear kernel was determined to be the most optimal support vector machine model. It should be noted that the linear 
SVM closely resembles logistic regression in its utilization of a linear plane to serve as the decision boundary.

#### Metrics
##### Classification Metrics
Accuracy | Sensitivity | Specificity | F1 Score | Precision | Recall | AUC  
:--: | :--: | :--: | :--: | :--: | :--: | :--:  
0.93171 | 0.9375 | 0.928 | 0.91463 | 0.89286 | 0.89286 | 0.9836

##### ROC Curve
![alt text](https://github.com/cpnguyen5/clf-bcwd/blob/master/results/SVM_auc.png "ROC Curve")


### 2. Logistic Regression (LR)
The Logistic Regression classifier places a “best fit” line that minimizes the squared error or distance between the 
line and data. In other words, it attempts to have data as tightly as possible. Logistic regression is essentially linear 
regression between the log-odds of an outcome (categorical output) and features.

#### Pros
Logistic regression is a very strong classifier for linear decision surfaces due to its implementation of the “best fit” 
line that minimizes the distance between the line and data points. Performance is superior when classifying samples that 
are at the ends of the spectrum (class 0 or class 1). In other words, it’s very effective in classifying samples that 
clearly belong to a certain classification label. 

#### Cons
Logistic Regression has poor performance when classifying samples that are in the middle of the spectrum (logistic 
curve). In addition, it is very sensitive to outliers as outliers will skew the placement of the best fit line. Although 
the model performs exceptionally well on linear data, it’s performance is not as great for non-linear datasets. 

#### Parameters
Penalty | C | Solver | fit_intercept 
:--: | :--: | :--: | :--: 
l2 | 100 | newton-cg | True 

**Cross-Validated Grid Search Score**: 0.975

##### Model Coefficients
`y= -5.72 + 7.89 X`<sub>size uniformity</sub>` + 4.36 X`<sub>shape uniformity</sub>` + 1.32 X`<sub>bare nuclei</sub>` + 5.97 X`<sub>bland chromatin</sub>` + 2.21 X`<sub>normal nucleoli</sub>

**Intercept** = -5.72

**Slope (β)**:
Feature | β<sub>x</sub> |
:-- | :--: |
Size Uniformity | 7.89 |
Shape Uniformity | 4.36 |
Bare Nuclei | 1.32 |
Bland Chromatin | 5.98 |
Normal Nucleoli | 2.21 |

#### Metrics
##### Classification Metrics
Accuracy | Sensitivity | Specificity | F1 Score | Precision | Recall | AUC  
:--: | :--: | :--: | :--: | :--: | :--: | :--:  
0.94146 | 0.9375 | 0.944 | 0.92593 | 0.91463 | 0.9375 | 0.9823

##### ROC Curve
![alt text](https://github.com/cpnguyen5/clf-bcwd/blob/master/results/Logistic%20Regression_auc.png "ROC Curve")


### 3. Gaussian Naive Bayes (GNB)
The Gaussian Naïve Bayes classifier model applies Baye’s Theorem that centers on the “naïve” assumption of independence 
between every pair of features.  It essentially is a conditional probability model that weighs each feature’s 
weight/significance independently, based on how much the feature correlates with the class label/outcome.

#### Pros
Gaussian Naïve Bayes is a relatively fast classifier, with a linear ( O(N) ) run-time. Another benefit is that it 
requires simpler training compared to other classifier models as it only requires less training data to fit the model 
and trains each probability distribution independently. Gaussian Naïve Bayes is also better for datasets of 
high-dimensional spaces, making its model generalizable for future datasets.

#### Cons
A big con of Gaussian Naïve Bayes is that it’s a decent classifier and a bad estimator. Thus, users can only trust its 
classification output, but not probability estimates for classes (predict_proba()). In addition, the model relies on the 
assumption that its data has a strong naïve independence between features and Gaussian distribution.

#### Metrics
##### Classification Metrics
Accuracy | Sensitivity | Specificity | F1 Score | Precision | Recall | AUC  
:--: | :--: | :--: | :--: | :--: | :--: | :--:  
0.92683 | 0.9625 | 0.904 | 0.91124 | 0.86517 | 0.9625 | 0.97

##### ROC Curve
![alt text](https://github.com/cpnguyen5/clf-bcwd/blob/master/results/Gaussian%20Naive%20Bayes_auc.png "ROC Curve")

### 4. Random Forest (RF)
Random Forests are classifiers that conducts feature selection implicitly (ranking features based on overall impact) and 
constructs a multitude of decision trees during training.  The model uses a criterion (e.g. Gini impurity or entropy) 
for information gain in order to split the tree and create nodes. Internal/non-leaf nodes are an attribute that requires 
further splitting and leaf nodes are ultimately the endpoint, in which a classification label is obtained. It should be 
noted that random forest classifiers conducts bagging by sampling training samples with replacement (similar to 
bootstrapping) in order to reduce overfitting.

#### Pros
Random Forests classifiers are simple to understand and interpret its results. It is also tolerant of less processed 
data, requiring less preprocessing of the data prior to fitting or training. In addition, Random Forests are robust to 
bad data assumptions and can handle any data type (e.g. categorical or quantitative). Other benefits include the fact 
the model does not expect linear features and has the capacity to handle large and high-dimensional datasets. Lastly, it 
should be noted that Random Forest also implicitly conducts feature selection by ranking each feature based on its 
overall impact.

#### Cons
On the contrary, one big caveat to Random Forest classifiers is that they’re prone to overfitting. Thus, making it not 
generalizable outside the training set (cannot be effectively applied on future datasets). The classifier model is also 
greatly affected by outliers. Lastly, another con is that certain relationships of the dataset may be hard to learn, 
resulting in fairly complex decision trees.

#### Parameters
max_features | n_estimators | criterion 
:--: | :--: | :--:
3 | 9 | gini

**Cross-Validated Grid Search Score**: 0.981

A linear kernel was determined to be the most optimal support vector machine model. It should be noted that the linear 
SVM closely resembles logistic regression in its utilization of a linear plane to serve as the decision boundary.

#### Metrics
##### Classification Metrics
Accuracy | Sensitivity | Specificity | F1 Score | Precision | Recall | AUC  
:--: | :--: | :--: | :--: | :--: | :--: | :--:  
0.96585 | 1.0 | 0.944 | 0.95808 | 0.91954 | 1.0 | 0.9751

##### ROC Curve
![alt text](https://github.com/cpnguyen5/clf-bcwd/blob/master/results/Random%20Forest_auc.png "ROC Curve")
