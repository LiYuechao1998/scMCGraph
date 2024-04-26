import numpy as np
from scipy.stats import norm
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
def standardNormalization(mat):

    mat_normalized = (mat - np.mean(mat, axis=0)) / np.std(mat, axis=0)
    return mat_normalized
def affinity_matrix(diff, K=20, sigma=0.5):
    N = diff.shape[0]
    diff = (diff + diff.T) / 2
    np.fill_diagonal(diff, 0)
    sorted_columns = np.sort(diff, axis=0)
    means = np.mean(sorted_columns[:K + 1, :], axis=0) + np.finfo(float).eps
    Sig = (np.outer(means, means) / 3) * 2 + (diff / 3) + np.finfo(float).eps
    Sig[Sig <= np.finfo(float).eps] = np.finfo(float).eps
    densities = norm.pdf(diff, 0, sigma * Sig)
    W = (densities + densities.T) / 2
    return W
def Standardization_distance(expression):

    expression = expression.T
    expression_normalized = standardNormalization(expression)
    expression_normalized.fillna(0, inplace=True)
    dist_matrix = pdist(expression_normalized, metric='euclidean')
    similarity_matrix = squareform(dist_matrix)
    similarity_matrix = pd.DataFrame(similarity_matrix, index=expression.index, columns=expression.index)

    return similarity_matrix

def snf(X_list, K=2, T=2, alpha=0.5):

    S_list = []
    print("    for X in X_list:")
    for X in X_list:
        print("D = cdist(X, X)")
        D = cdist(X, X)
        print("D_sort = np.sort(D, axis=1)")
        D_sort = np.sort(D, axis=1)
        print("D_mean = np.mean(D_sort[:, 1:K+1], axis=1)")
        D_mean = np.mean(D_sort[:, 1:K+1], axis=1)
        print("S = 1 / (1 + D_mean.reshape(-1, 1))")
        S = 1 / (1 + D_mean.reshape(-1, 1))
        print("S_list.append(S)")
        S_list.append(S)

    print("F = np.mean(S_list, axis=0)")
    F = np.mean(S_list, axis=0)
    print("for t in range(T):")
    for t in range(T):
        print("F = alpha * np.dot(F, F.T) + (1 - alpha) * np.mean(S_list, axis=0)")
        F = alpha * np.dot(F, F.T) + (1 - alpha) * np.mean(S_list, axis=0)

    return F
def Matrix_fusion(fintruefeatures,W):
    print("fintruefeatures = Standardization_distance(fintruefeatures)")
    fintruefeatures = Standardization_distance(fintruefeatures)
    print("W = Standardization_distance(W)")
    W = Standardization_distance(W)
    fintruefeatures_np = fintruefeatures.values
    print(fintruefeatures_np)
    fintruefeatures_np = affinity_matrix(fintruefeatures_np, K=10, sigma=0.5)
    print(fintruefeatures_np)

    W_np = W.values
    print(W_np)
    W_np = affinity_matrix(W_np, K=10, sigma=0.5)
    print(W_np)
    try:
        F = snf([fintruefeatures_np, W_np])
        print(F)
    except Exception as e:
        F = (fintruefeatures_np + W_np) / 2
    return F

def ssfm(adj_matrices, k=3):


    eig_values = []
    for adj_matrix in adj_matrices:
        eig_values.append(np.linalg.eigvals(adj_matrix))
    eig_values = np.array(eig_values)


    dist_matrix = np.zeros((len(adj_matrices), len(adj_matrices)))
    for i in range(len(adj_matrices)):
        for j in range(len(adj_matrices)):
            dist_matrix[i][j] = np.linalg.norm(eig_values[i] - eig_values[j])

    sim_matrix = np.exp(-dist_matrix)


    fused_matrix = np.zeros(adj_matrices[0].shape)
    for i in range(fused_matrix.shape[0]):
        for j in range(fused_matrix.shape[1]):
            values = []
            for adj_matrix in adj_matrices:
                values.append(adj_matrix[i][j])
            fused_matrix[i][j] = np.sum(sim_matrix.dot(values)) / np.sum(sim_matrix)

    flattened = fused_matrix.flatten()
    indices = np.argsort(flattened)[::-1][:k]
    flattened[indices] = 1
    flattened[flattened != 1] = 0

    adj_matrix = flattened.reshape(fused_matrix.shape)

    return adj_matrix


def ssfm_new(adj_matrices, k=3):

    eig_values = np.array([np.linalg.eigvals(adj_matrix) for adj_matrix in adj_matrices])


    dist_matrix = np.linalg.norm(eig_values[:, None] - eig_values, axis=-1)

    sim_matrix = np.exp(-dist_matrix)

    fused_matrix = np.sum(sim_matrix[:, :, None] * np.array(adj_matrices), axis=0) / np.sum(sim_matrix)

    indices = np.argpartition(fused_matrix.flatten(), -k)[-k:]
    flattened = np.zeros_like(fused_matrix.flatten())
    flattened[indices] = 1
    adj_matrix = flattened.reshape(fused_matrix.shape)
    return adj_matrix

def summ(adj_matrices, k=3):

    fused_matrix = np.sum(adj_matrices, axis=0)
    indices = np.argpartition(fused_matrix.flatten(), -k)[-k:]
    flattened = np.zeros_like(fused_matrix.flatten())
    flattened[indices] = 1
    adj_matrix = flattened.reshape(fused_matrix.shape)
    return adj_matrix