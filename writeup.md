# Results
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

