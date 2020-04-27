#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
import scipy
from matplotlib import pyplot as plt

import seaborn as sns
sns.set(color_codes=True)
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        # Choose the most common label among data points in the cluster
        new_label=scipy.stats.mode(real_labels[idx])[0][0]
        permutation.append(new_label)
    return permutation

def split(word): 
    return [char for char in word]  

def toint(x):
    dic={"A":0,"C":1,"G":2,"T":3}
    res=[]
    for ch in split(x):
         res.append(dic[ch])
         aux=dic[ch]
    if len(x) != 1:
        return res
    else: 
        return aux
    
def get_features_and_labels(filename):
    df= pd.read_csv(filename, sep="\t")
    features=[]
    X = df.loc[:, "X"].values
    for el in X:
        features.append(toint(el))
    features = np.asarray(features)
    labels= df.y.values
    #labels = list(map(int, labels))
    return (features, labels)

def plot(distances, method='average', affinity='euclidean'):
    mylinkage = hc.linkage(sp.distance.squareform(distances), method=method)
    g=sns.clustermap(distances, row_linkage=mylinkage, col_linkage=mylinkage )
    g.fig.suptitle(f"Hierarchical clustering using {method} linkage and {affinity} affinity")
    plt.show()

def cluster_euclidean(filename):
    features,labels=get_features_and_labels(filename)
    model =AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage="average")
    clustering=model.fit(features)
    permutation3 = find_permutation(2, labels, model.labels_)
    acc = accuracy_score(labels, [ permutation3[label] for label in model.labels_])

    return acc

def cluster_hamming(filename):
    features,labels=get_features_and_labels(filename)
    paird=pairwise_distances(features,metric="hamming")


    
    model =AgglomerativeClustering(n_clusters=2,affinity='precomputed',linkage="average")
    clustering=model.fit(paird)
    permutation3 = find_permutation(2, labels, model.labels_)
    acc = accuracy_score(labels, [ permutation3[label] for label in model.labels_])

    return acc


def main():
    print("Accuracy score with Euclidean affinity is", cluster_euclidean("src/data.seq"))
    print("Accuracy score with Hamming affinity is", cluster_hamming("src/data.seq"))
    print(toint("A"))
    #print(get_features_and_labels( "src/data.seq"))
if __name__ == "__main__":
    main()
