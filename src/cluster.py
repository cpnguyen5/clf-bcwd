import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    scaler = MinMaxScaler().fit(data_array) # scaler object fitted to training set

    # Transform
    scaled_data = scaler.transform(data_array)
    return (scaled_data, scaler)


def kmeans(scaler, norm_data, k):
    """
    The function creates an instance of sklearn.cluster.KMeans with the indicated n_clusters. The model is fitted with
    normalized data to obtain the cluster labels, which is subsequently assigned to the original data. The assignment
    of cluster labels is for appropriate domain interpretation of plots. The silhouette score indicating the performance
    of the clustering is returned in addition to the model and a list of the group of original data by their clusters.

    :param scaler: scaler object
    :param norm_data: normalized data
    :param k: number of clusters
    :return: tuple of (list of arrays of original data for each cluster/label, silhouette score, model)
    """
    #Apply K-Means clustering on normalized data to obtain labels
    model = KMeans(n_clusters=k) #instance of k-means clustering model
    model = model.fit(norm_data) #Fit model to normalized data to provide cluster labeling of data
    n_clusters = model.n_clusters #number of clusters
    labels = model.labels_ #cluster labels based on normalized data

    orig_data = scaler.inverse_transform(norm_data)

    #Filter original data by cluster labels fitted from normalized data
    lst_orig = [] #Set up accumulator for arrays of clusters for original data
    for i in range(n_clusters):
        cluster_array = orig_data[labels==i] #Filter original data for specified cluster label from norm data
        lst_orig += [cluster_array] #Accumulate filtered array of specified cluster label

    sil_score = silhouette_score(norm_data, labels) #Silhouette Score
    return (lst_orig, sil_score, model)


def agglom_clust(scaler, norm_data, n_clusters, affinity, linkage):
    """
    The function creates an instance of sklearn.cluster.AgglomerativeClustering with the indicated n_clusters. The model
    is fitted with normalized data to obtain the cluster labels, which is subsequently assigned to the original data.
    The assignment of cluster labels is for appropriate domain interpretation of plots. The silhouette score indicating
    the performance of the clustering is returned in addition to the model and a list of the group of original data by
    their clusters.

    Please note that the linkage criterion, "ward," only accepts the metric of "euclidean" (affinity).

    :param scaler: scaler object
    :param norm_data: normalized data
    :param n_clusters: number of clusters
    :param affinity: affinity metric to compute linkage (euclidean or manhattan)
    :param linkage: linkage criterion (ward, complete, average)
    :return: tuple of (list of arrays of original data for each cluster/label, silhouette score, mode)
    """
    #Apply Agglomerative Clustering on normalized data to obtain labels
    model = AgglomerativeClustering(n_clusters = n_clusters, affinity = affinity, linkage = linkage) #instance of Agglomerative Clustering model
    model = model.fit(norm_data) #Fit model to normalized data to provide cluster labeling of data
    num_clusters = model.n_clusters #number of clusters
    labels = model.labels_ #cluster labels based on normalized data

    orig_data = scaler.inverse_transform(norm_data)

    #Filter original data by cluster labels fitted from normalized data
    lst_orig = [] #Set up accumulator for arrays of cluster for original data
    for i in range(num_clusters):
        cluster_array = orig_data[labels==i] #Filter original data for specified cluster label from norm data
        lst_orig += [cluster_array] #Accumulate filtered array of specified cluster label

    sil_score = silhouette_score(norm_data, labels) #Silhouette Score
    return (lst_orig, sil_score, model)



#main
if __name__ == '__main__':
    np.random.seed(2)

    # Obtain Data
    path = getPath()
    data = readCSV(path)
    data.drop('class', axis=1, inplace=True)
    scaled_data, scaler_obj = scale(data.iloc[:,1:3])

    # Cluster
    km_lst_orig, km_sil_score, km_model = kmeans(scaler_obj,  scaled_data, k=2)

