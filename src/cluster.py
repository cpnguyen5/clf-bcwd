import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


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


def scale(DataFrame):
    """
    Function takes a pandas DataFrame, converts it into a NumPy array, and returns the scaled versions.

    :param DataFrame: pandas DataFrame of data
    :return: tuple(normalized array, scaler objecT)
    """
    # Convert to NumPy array
    data_array = DataFrame.as_matrix()

    # Fit scaler
    scaler = StandardScaler().fit(data_array) # scaler object fitted to training set

    # Transform
    scaled_data = scaler.transform(data_array)
    return (scaled_data, scaler)


def pca(data, n_components=None):
    """
    Function performs dimensional reduction of features into principal components using sklearn.decomposition.PCA
    class. The PCA-input data is expected to have a zero mean and unit variance distribution.

    :param data: scaled data
    :param n_components: number of principal components
    :return: tuple of (PCA_trans_data, pca_object)
    """
    pca = PCA(n_components = n_components) #PCA object
    pca.fit(data)
    components = pca.transform(data) #transformed data
    return (components, pca)


def kmeans(pca_obj, components, k):
    """
    The function creates an instance of sklearn.cluster.KMeans with the indicated n_clusters. The model is fitted with
    normalized data to obtain the cluster labels. Function generates a plot of the clustering and returns the silhouette
    score.

    :param norm_data: normalized data
    :param k: number of clusters
    :return: silhouette score
    """
    #Apply K-Means clustering on normalized data to obtain labels
    model = KMeans(init='k-means++', n_clusters=k) #instance of k-means clustering model
    model.fit(components) #Fit model to normalized data to provide cluster labeling of data
    n_clusters = model.n_clusters #number of clusters
    labels = model.labels_ #cluster labels
    var = pca_obj.explained_variance_ratio_

    # Obtain cluster groups
    lst_clusters = []
    for i in range(n_clusters):
        lst_clusters += [components[labels==i]]

    sil_score = silhouette_score(components, labels) #Silhouette Score

    # Plot
    fig = plt.figure()
    # plt.scatter(components[:, 0], components[:, 1], c=labels, label=labels)
    plt.scatter(lst_clusters[0][:, 0], lst_clusters[0][:, 1], c='b', label='cluster 0')
    plt.scatter(lst_clusters[1][:, 0], lst_clusters[1][:, 1], c='r', label='cluster 1')
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.title('PCA Clustering ({:.2f}% Var. Explained)'.format(var.sum() * 100))
    plt.legend(loc='upper right')
    plt.show()
    plt.close(fig)

    # Save Plot
    abspath = os.path.abspath(__file__)  # absolute pathway to file
    head_path, f_name = os.path.split(abspath)
    work_dir = os.path.split(head_path)[0]  # root working dir

    aucfig_path = os.path.join(work_dir, 'results', 'kmeans.png')
    fig.savefig(aucfig_path, format='png')
    return sil_score


#main
if __name__ == '__main__':
    np.random.seed(2)

    # Obtain Data
    path = getPath()
    data = readCSV(path)
    X = data.drop(['id','class'], axis=1)
    scaled_X, scaler_obj = scale(X)

    # PCA
    components, pca_obj = pca(scaled_X, n_components=2)
    print "PCA (Dimension Reduction):"
    print "================================="
    print "Complete Principle Components"
    print "---------------------------------"
    components_all, pca_obj_all = pca(scaled_X, n_components=None)
    print "Explained Variance Shape: ", pca_obj_all.explained_variance_.shape
    print "Explained Variance: \n", pca_obj_all.explained_variance_
    print "Explained Variance Ratio: \n", pca_obj_all.explained_variance_ratio_
    print
    print "MLE"
    print "---------------------------------"
    components_mle, pca_obj_mle = pca(scaled_X, n_components='mle')
    print "Explained Variance Shape: ", pca_obj_mle.explained_variance_.shape
    print "Explained Variance: = ", pca_obj_mle.explained_variance_
    print "Explained Variance Ratio = ", pca_obj_mle.explained_variance_ratio_
    print
    print "2 Principle Components"
    print "---------------------------------"
    print "Explained Variance Shape: ", pca_obj.explained_variance_.shape
    print "Explained Variance: = ", pca_obj.explained_variance_
    print "Explained Variance Ratio = ", pca_obj.explained_variance_ratio_
    print

    # Cluster
    sil = kmeans(pca_obj, components, k=2)
    components, pca_obj = pca(scaled_X, n_components=2)
    print "Clustering"
    print "================================="
    print "Silhouette Score = %.3f" % sil