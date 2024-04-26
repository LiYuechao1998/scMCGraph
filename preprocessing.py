import numpy as np
import scipy.sparse as sp

def preprocess_graph(adjs):
    numView = len(adjs)
    adjs_normarlized = []
    for v in range(numView):
        adj = sp.coo_matrix(adjs[v])
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
        adjs_normarlized.append(adj_normalized.tolist())
    return np.array(adjs_normarlized)


