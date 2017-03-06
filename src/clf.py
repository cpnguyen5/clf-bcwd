import os, sys


def getPath():
    """
    Function takes no parameters, returning the pathway of the data csv file.

    :return: directory pathway of csv file
    """
    abspath = os.path.abspath(__file__) # absolute pathway to file
    head_path, f_name = os.path.split(abspath)
    work_dir = os.path.split(head_path)[0] # root working dir
    csvpath = os.path.join(work_dir, 'data', 'breast-cancer-wisconsin.csv')
    return csvpath

