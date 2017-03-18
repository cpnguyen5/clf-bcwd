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
    :return: normalized array
    """
    # Convert to NumPy array
    data_array = DataFrame.as_matrix()

    # Fit scaler
    scaler = MinMaxScaler().fit(data_array) # scaler object fitted to training set

    # Transform
    scaled_data = scaler.transform(data_array)
    return scaled_data


#main
if __name__ == '__main__':
    np.random.seed(2)

    # Obtain Data
    path = getPath()
    data = readCSV(path)
    data.drop('class', axis=1, inplace=True)
    scaled_data = scale(data)
