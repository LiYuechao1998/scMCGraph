import os

import numpy as np
import scanpy as sc
import pandas as pd
from pathway import get_pathway_W, pathwayname
from Batch_removal import Batch_removal
from Matrix_fusion import Matrix_fusion, ssfm, ssfm_new,summ
from PPMI import diffusion_fun_improved_ppmi_dynamic_sparsity
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from snf_function import get_snf_matrix, get_knn_graph
from scipy.sparse import save_npz, load_npz
# import dask.dataframe as dd



def load_data(Traindataname_truefeatures, Traindataname_truelabels, Testdataname_truefeatures, Testdataname_truelabels, pathwaytype):
    Traindataname_truefeatures = sc.read_csv(Traindataname_truefeatures)
    Traindataname_truefeatures = Traindataname_truefeatures.transpose()
    Traindataname_truelabels = pd.read_csv(Traindataname_truelabels, index_col=0)
    Testdataname_truefeatures = sc.read_csv(Testdataname_truefeatures)
    Testdataname_truefeatures = Testdataname_truefeatures.transpose()
    Testdataname_truelabels =  pd.read_csv(Testdataname_truelabels, index_col=0)



    Traindataname_fintruefeatures, Testdataname_fintruefeatures, Traindataname_merged_harmony_data, Testdataname_merged_harmony_data, Traindataname_truefeatures_obs_df, Testdataname_truefeatures_obs_df = Batch_removal(Traindataname_truefeatures, Traindataname_truelabels, Testdataname_truefeatures, Testdataname_truelabels)
    print("Traindataname_truefeatures_obs_df")
    print(Traindataname_truefeatures_obs_df)

    data_len = len(Traindataname_truefeatures_obs_df)
    labels = pd.concat([Traindataname_truefeatures_obs_df, Testdataname_truefeatures_obs_df], axis=0)
    one_hot_labels = pd.get_dummies(labels['cell_type'])
    Train_one_hot_labels = one_hot_labels.iloc[:data_len, :]
    Test_one_hot_labels = one_hot_labels.iloc[data_len:, :]
    label_number = one_hot_labels.shape[1]
    Train_one_hot_labels_values = Train_one_hot_labels.values
    Test_one_hot_labels_values = Test_one_hot_labels.values





    Traindataname_fintruefeatures_transpose = Traindataname_fintruefeatures.transpose()
    Testdataname_fintruefeatures_transpose = Testdataname_fintruefeatures.transpose()
    k_train = Traindataname_merged_harmony_data.shape[0]//100
    k_test = Testdataname_merged_harmony_data.shape[0]//100

    # k_train = Traindataname_merged_harmony_data.shape[0]//50
    # k_test = Testdataname_merged_harmony_data.shape[0]//50


    # k_train = Traindataname_merged_harmony_data.shape[0]//20
    # k_test = Testdataname_merged_harmony_data.shape[0]//20
    # k_train = Traindataname_merged_harmony_data.shape[1]//200
    print("Traindataname_merged_harmony_data.shape[0]")
    print(Traindataname_merged_harmony_data.shape[0])
    print("k_train")
    print(k_train)

    print("Testdataname_merged_harmony_data.shape[0]")
    print(Testdataname_merged_harmony_data.shape[0])
    print("k_test")
    print(k_test)



    if pathwaytype == 'C:/Users/User/Desktop/pathway/cross_omics':
        print("pathwaytype == 'cross_omics")
        folder_path = 'C:/Users/User/Desktop/pathway/cross_omics/'
        pathway = pathwayname(folder_path)
        print(len(pathway))
        Train_rownetworks = np.empty((0, Traindataname_merged_harmony_data.shape[0], Traindataname_merged_harmony_data.shape[0]))
        Test_rownetworks = np.empty((0, Testdataname_merged_harmony_data.shape[0], Testdataname_merged_harmony_data.shape[0]))
        for i in range(len(pathway)):
            pathway_name = folder_path + pathway[i]
            name = pathway[i].split('.')[0]
            print("name------------------------------------------")
            print(name)

            if os.path.exists('C:/Users/User/Desktop/data/cross_species_brain/brain/train_w_{}.csv'.format(name)):
                train_w_csv = pd.read_csv('C:/Users/User/Desktop/data/cross_species_brain/brain/train_w_{}.csv'.format(name))
                train_w_np = np.load('C:/Users/User/Desktop/data/cross_species_brain/brain/train_w_{}.npy'.format(name))
                train_w = train_w_csv
            else:
                print("--------------")
                continue


            if os.path.exists('C:/Users/User/Desktop/data/cross_species_brain/brain/test_w_{}.csv'.format(name)):
                test_w_csv = pd.read_csv('C:/Users/User/Desktop/data/cross_species_brain/brain/test_w_{}.csv'.format(name))
                test_w_np = np.load('C:/Users/User/Desktop/data/cross_species_brain/brain/test_w_{}.npy'.format(name))
                test_w = test_w_csv
            else:
                continue
            ##############################
            #######       snf    #########
            ##############################
            # print("train_G = get_snf_matrix(Traindataname_fintruefeatures_transpose, train_w, k_train)")
            train_G = get_snf_matrix(Traindataname_fintruefeatures_transpose, train_w, k_train)
            save_npz('C:/Users/User/Desktop/data/cross_species_brain/brain/train_G_{}.npz'.format(name), train_G)

            print("train_G = train_G.A")
            train_G = train_G.A
            print(train_G)
            print("Train_rownetworks = np.append(Train_rownetworks, [train_G], axis=0)")
            Train_rownetworks = np.append(Train_rownetworks, [train_G], axis=0)
            test_G = get_snf_matrix(Testdataname_fintruefeatures_transpose, test_w, k_test)
            save_npz('C:/Users/User/Desktop/data/cross_species_brain/brain/test_G_{}.npz'.format(name), test_G)

            test_G = test_G.A
            Test_rownetworks = np.append(Test_rownetworks, [test_G], axis=0)

    else:
        print("pathwaytype unknown")
    print("Train_A_representation_matrix")
    print(Train_rownetworks.shape)
    print(Train_rownetworks)

    Train_A_representation_matrix = ssfm(Train_rownetworks, k=k_train)
    print(Train_A_representation_matrix)
    np.save('C:/Users/User/Desktop/data/cross_species_brain/brain/train_ssfm_matrix.npy', Train_A_representation_matrix)

    print("Test_A_representation_matrix")

    Test_A_representation_matrix = ssfm(Test_rownetworks, k=k_test)
    print(Test_A_representation_matrix)
    np.save('C:/Users/User/Desktop/2data/cross_species_brain/brain/test_ssfm_matrix.npy', Test_A_representation_matrix)

    Train_P = diffusion_fun_improved_ppmi_dynamic_sparsity(Train_A_representation_matrix, path_len=2, k=1.0)
    Train_P_A = Train_P.A
    print("Train_P_A")
    Train_rownetworks = np.append(Train_rownetworks, [Train_P_A], axis=0)
    Test_P = diffusion_fun_improved_ppmi_dynamic_sparsity(Test_A_representation_matrix, path_len=2, k=1.0)
    Test_P_A = Test_P.A
    print("Test_P_A")
    Test_rownetworks = np.append(Test_rownetworks, [Test_P_A], axis=0)

    Train_numView = Train_rownetworks.shape[0]
    print(Train_numView)
    Test_numView = Test_rownetworks.shape[0]
    print(Test_numView)
    return np.array(Train_rownetworks), np.array(Test_rownetworks), Train_numView, Test_numView, Traindataname_merged_harmony_data, Testdataname_merged_harmony_data, Train_one_hot_labels_values, Test_one_hot_labels_values, label_number