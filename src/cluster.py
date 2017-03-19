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
    of the clustering is returned in addition to a string indicating the model and a list of the group of original data
    by their clusters.

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
    return (lst_orig, sil_score, "kmeans")


def agglom_clust(scaler, norm_data, n_clusters, affinity, linkage):
    """
    The function creates an instance of sklearn.cluster.AgglomerativeClustering with the indicated n_clusters. The model
    is fitted with normalized data to obtain the cluster labels, which is subsequently assigned to the original data.
    The assignment of cluster labels is for appropriate domain interpretation of plots. The silhouette score indicating
    the performance of the clustering is returned in addition to a string indicating the model and a list of the group
    of original data by their clusters.

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
    return (lst_orig, sil_score, "agglomerative")


def plot(cluster_tuple, affin = None, linkage = None):
    """
    This function takes two required parameters: cluster_tuple and ex; and two optional parameters: affin and linkage.
    Cluster_tuple are the outputs of the kmeans or agglom_clust functions. The optional parameters correlate to the
    affinity and linkage parameters of the sklearn.clustering.AgglomerativeClustering function and is used in this
    function for labeling purposes.
    Given these inputs, this function will use matplotlib to plot the data, identifying each cluster assignment with
    its respective unique shape and color.

    :param cluster_tuple: Output of kmeans or agglom_clust function containing list of original data arrays for each
    cluster label, silhouette score, and string indicating clustering algorithm.
    :param ex: string containing "ex#" to indicate directory of specific data file
    :param affin: Specified affinity or metric to compute linkage
    :param linkage: Specified linkage criterion
    :return: Plot of clustering assignments
    """
    plt.figure()#automatically increase Figure number
    lst_clusters = cluster_tuple[0] #list of arrays for each cluster label
    n_clusters = len(lst_clusters) #number of clusters
    sil_score = cluster_tuple[1] #silhouette score
    clust0 = lst_clusters[0] #array for cluster 0 (label 0)
    clust1 = lst_clusters[1] #array for cluster 1 (label 1)

    if n_clusters < 3: #Condition for number of clusters less than 3
        if clust0.shape[1] < 2: #Condition for 1D arrays
            #Create array of 0-valued elements to specify y = 0 for plotting purposes
            y0 = np.zeros(clust0.shape)
            y1 = np.zeros(clust1.shape)

            #Scatter plot for each cluster assignment
            plt.scatter(clust0[:, [0]], y0, c = 'b', marker = 'x') #Color blue, marker x
            plt.scatter(clust1[:, [0]], y1, c= 'r', marker = 'o') #Color red, marker o
        else:
            plt.scatter(clust0[:, [0]], clust0[:, [1]], c = 'b', marker = 'x')
            plt.scatter(clust1[:, [0]], clust1[:, [1]], c= 'r', marker = 'o')
        plt.legend(('cluster 0', 'cluster 1'), loc = "lower right")

    else:
        clust2 = lst_clusters[2] #array for cluster 2 (label 2)
        if clust0.shape[1] < 2: #Condition for 1D arrays
            #Create array of 0-valued elements to specify y = 0 for plotting purposes
            y0 = np.zeros(clust0.shape)
            y1 = np.zeros(clust1.shape)
            y2 = np.zeros(clust2.shape)

            #Scatter plot for each cluster assignment
            plt.scatter(clust0[:, [0]], y0, c = 'b', marker = 'x') #Color blue, marker x
            plt.scatter(clust1[:, [0]], y1, c= 'r', marker = 'o') #Color red, marker o
            plt.scatter(clust2[:, [0]], y2, c= 'y', marker = '^') #Color yellow, marker triangle
        else:
            plt.scatter(clust0[:, [0]], clust0[:, [1]], c = 'b', marker = 'x')
            plt.scatter(clust1[:, [0]], clust1[:, [1]], c= 'r', marker = 'o')
            plt.scatter(clust2[:, [0]], clust2[:, [1]], c = 'y', marker = '^')
        plt.legend(('cluster 0', 'cluster 1', 'cluster 2'), loc = "lower right")

    #Condition to Label Title of each Figure/plot with respective Clustering Algorithms, Sihouette Score, Data
    if cluster_tuple[2] == "kmeans":
        plt.suptitle("K-Means Clustering \n Original Data (Normalized Labels) -- Silhouette Score: %0.5f" % (sil_score))
    else:
        plt.suptitle("Agglomerative Clustering: %s, %s \n Original Data (Normalized Labels) -- Silhouette Score: %0.5f" %
                     (affin, linkage, sil_score))


#main
if __name__ == '__main__':
    np.random.seed(2)

    # Obtain Data
    path = getPath()
    data = readCSV(path)
    data.drop('class', axis=1, inplace=True)
    scaled_data, scaler_obj = scale(data.iloc[:,1:3])

    # Cluster
    km_tuple = kmeans(scaler_obj,  scaled_data, k=2)
    km_lst_orig, km_sil_score, km_model = km_tuple
    plot(km_tuple)
    plt.show()
    plt.close()