import snf
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


def get_knn_graph(arr, knn):

    m, n = arr.shape
    out = np.zeros((m, n))
    for i in range(m):
        row = arr[i]
        top_k = np.argpartition(-row, knn)[:knn]
        out[i, top_k] = 1
    for i in range(min(m, n)):
        out[i, i] = 1
    return out

def get_snf_matrix(features_graph_pathway, features_graph, knn):
    features_graph_pathway = features_graph_pathway.T
    features_graph = features_graph.T
    if features_graph.size == 0:

        print(features_graph_pathway.shape)
        features_graph_all = [features_graph_pathway]
        affinity_networks = snf.make_affinity(features_graph_all, metric='euclidean', K=20, mu=0.5)
        fused_network = get_knn_graph(affinity_networks[0], knn)
        sparse_matrix = csr_matrix(fused_network)
        return sparse_matrix

    features_graph_all = [features_graph_pathway, features_graph]

    affinity_networks = snf.make_affinity(features_graph_all, metric='euclidean', K=20, mu=0.5)

    fused_network = snf.snf(affinity_networks, K=20, t=1, alpha=1.0)

    fused_network = get_knn_graph(fused_network, knn)

    '''
    graph2 = kneighbors_graph(features_graph, knn, mode='connectivity', include_self=True)
    graph2 = graph2.toarray()
    fused_network = fused_network + graph2
    '''

    sparse_matrix = csr_matrix(fused_network)

    # sparse_matrix = kneighbors_graph(fused_network, knn, mode='connectivity', include_self=True)
    return sparse_matrix

