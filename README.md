# Classification: Breast Cancer Wisconsin Diagnostic Data
The project explores machine learning approaches, specifically supervised learning (classification), to create
a predict model identifying the diagnoses of breast cancer.

## Dataset
The Breast Cancer Wisconsin Diagnostic dataset is sourced from the University of California, Irvine's machine 
learning repository. For more, information please reference the site's [documentation].  
[documentation]: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

The dataset has 16 instances of missing attributes, denoted by `?` in the original data.  

#### Features
The data contains 10 features related to the diagnoses of breast cancer:
  1. **Sample Code Number**: `id`
  2. **Clump Thickness**: `clump_thickness`
  3. **Uniformity of Cell Size**: `size_uniformity`
  4. **Uniformity of Cell Shape**: `shape_uniformity`
  5. **Marginal Adhesion**: `adhesion`
  6. **Single Epithelial Cell Size**: `epithelial_size`
  7. **Bare Nuclei**: `bare_nuclei`
  8. **Bland Chromatin**: `bland_chromatin`
  9. **Normal Nucleoli**: `normal_nucleoli`
  10. **Mitoses**: `mitoses`


#### Labels
The classes for the dataset include: benign for non-cancerous tumors and malignant for active cancer tumors.
  1. **Benign**: `0`
  2. **Malignant**: `1`
  
There are approximately 65% benign cases (n<sub>benign</sub>=658) and 34.5% malignant cases (n<sub>malignant</sub>=241).
  

## Modeling
### Running Code
```
Run using the following command:
    
  python src/clf.py
```

### Dataset Partition
Training Set | Testing Set | Random State (Generator)
:---: | :---: | :---:
70% (n=478) | 30% (n=205) | 2

**Note**: The random seed generator is implemented for developmental purposes to ensure consistency of data partitioning
as the code is debugged and tweaked.

### Feature Selection
Feature selection, otherwise known as the reduction of dimensionality on sample sets, is intended to
improve the estimator's accuracy and boost performance, especially on very high-dimensional datasets.  

Essentially, it's implemented as a *pre-processing* step to fitting a predictive model (supervised learning) with the 
purpose of reducing noisy, insignificant features.

Two feature selection methods were explored: **univariate feature selection** and **Random Forest**.
#### Univariate Feature Selection
Univariate feature selection selects the best features based on univariate statistical tests.

##### Function
The`sklearn.feature_selection.SelectKBest` function will be utilized for univariate feature selection. It removes all but
**`k`** highest scoring features (*per their scoring function scores*).  

##### Scoring Function
**Chi<sup>2</sup>** is the scoring function used for this univariate feature selection. The chi<sup>2</sup> statistic 
 between each non-negative feature and class is computed.
 
 The chi<sup>2</sup> test measures dependence between variables, filtering out features most likely to be independent
 of class (lower chi<sup>2</sup> scores and higher p-values). Thus, irrelevant for classification.

#### Random Forest - Feature Importances
The Random Forest classifier model also implicitly conducts feature selection during its fitting/training. As the estimator
prunes the decision trees into smaller and more optimal decision trees, the features are ranked by importance. Thus, 
enabling the recognition of the most important features from the relatively insignificant ones in order to boost the 
classifier's performance.

The ranking of features are an attribute of the model and can be viewed through the attribute: `.feature_importances_`

### Models
1. Support Vector Machine
2. Logistic Regression
3. Gaussian Naive Bayes
4. Random Forest


### Results
Please refer to the **writeup.md** markdown file for discussion of the results.

#### Metrics
