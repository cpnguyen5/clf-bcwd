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
  
There are approximately 65% benign cases (n<sub>benign</sub>=658) and 34.5% malignant cases (n<sub>malignant>/sub>=241).
  

