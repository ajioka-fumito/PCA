#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:15:08 2019

@author: ajioka fumito
"""

from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# create dataset
def create_X():
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns = boston.feature_names)
    data = data.values
    X = np.delete(data,12,1)
    mean = np.mean(X,axis=0)
    std  = np.std(X,axis=0) 
    X = (X-mean)/std
    return X

def create_t():
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns = boston.feature_names)
    data = data.values
    t = data[:,12]
    mean = np.mean(t,axis=0)
    std  = np.std(t,axis=0) 
    t = (t-mean)/std
    return t

# useing sklearn return components & transform_X
def PCA_Sk(X,n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    x = pca.transform(X)
    components = pca.components_
    return components,x

# using numpy return compnonts & transform_X
def PCA_Sc(X,n_components):
    columns = X.shape[1]
    # calucurate X_TX
    X_TrX = np.dot(X.T,X)
    # clucurate eigenvalue & Eigenvector
    w,v = np.linalg.eig(X_TrX)
    # create D
    D = []
    for i in range(n_components):
        D.append(v[:,i])
    D = np.array(D)
    D_reshape = np.reshape(D,(n_components,columns))
    # colucurate D_TX
    X_transform = np.dot(D_reshape,X.T)
    return X_transform
        
if __name__ =='__main__':
    # create dataset
    X = create_X()
    # sklearn aproach
    components,x = PCA_Sk(X,2)
    P_sk = np.dot(components,X.T)
    # scratch aproarch
    P_Sc = PCA_Sc(X,2)
