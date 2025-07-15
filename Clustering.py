'''
Multi-view clustering and evaluation in MvCLN (CVPR2021)
'''

import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from munkres import Munkres
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def Clustering(x_list, y):
    # logging.info('******** Clustering ********')
    n_clusters = np.size(np.unique(y))

    # np.random.seed(1)
    x_final_concat = np.concatenate(x_list[:], axis=1)
    # x_final_concat=x_list
    kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10})

    y_preds = get_y_preds(y, kmeans_assignments, n_clusters)
    if np.min(y) == 1:
        y = y - 1
    accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity = clustering_metric(y, kmeans_assignments, n_clusters)
    score=dict({'ACC': accuracy, 'NMI': nmi, 'ARI': ari,'f_score':f_score,'f_score_weighted':f_score2,'precision':precision,'precision_weighted':precision2,'recall':recall,'purity':purity})

    return y_preds,score,accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred




def clustering_metric(y_true, y_pred, n_clusters, verbose=False, decimals=4):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    accuracy = metrics.accuracy_score(y_true, y_pred_ajusted)
    accuracy = np.round(accuracy, decimals)
    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred_ajusted)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred_ajusted)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred_ajusted)
    ari = np.round(ari, decimals)
    # fscore
    f_score = metrics.f1_score(y_true, y_pred_ajusted, average='macro')
    f_score = np.round(f_score, decimals)
    f_score2 = metrics.f1_score(y_true, y_pred_ajusted, average='weighted')
    f_score2 = np.round(f_score2, decimals)
    # precision
    precision = metrics.precision_score(y_true, y_pred_ajusted, average='macro')
    precision = np.round(precision, decimals)
    precision2 = metrics.precision_score(y_true, y_pred_ajusted, average='weighted')
    precision2 = np.round(precision2, decimals)
    # recall
    recall = metrics.recall_score(y_true, y_pred_ajusted, average='macro')
    recall = np.round(recall, decimals)
    # Purity
    purity = Purity(y_true, y_pred_ajusted)
    purity = np.round(purity, decimals)

    return accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity

def Purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)
def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    '''
    Using either a newly instantiated ClusterClass or a provided
    cluster_obj, generates cluster assignments based on input data

    x:              the points with which to perform clustering
    cluster_obj:    a pre-fitted instance of a clustering class
    ClusterClass:   a reference to the sklearn clustering class, necessary
                    if instantiating a new clustering class
    n_clusters:     number of clusters in the dataset, necessary
                    if instantiating new clustering class
    init_args:      any initialization arguments passed to ClusterClass

    returns:    a tuple containing the label assignments and the clustering object
    '''
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(x)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_assignments, cmap='viridis')
    # plt.title("KMeans Clustering with t-SNE Visualization")
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.colorbar(label='Cluster Label')
    # plt.show()
    return cluster_assignments, cluster_obj
