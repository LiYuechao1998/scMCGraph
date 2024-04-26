import pandas as pd
import numpy as np
import os

def pathwayname(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path):
            file_names.append(file_name)
    return file_names
def subsetGeneSets(gene_sets, genes):
    subset = {}
    for name, data in gene_sets.items():
        gene_list = data["genes"]
        subset_genes = [gene for gene in genes if gene in gene_list]
        if subset_genes:
            subset[name] = {"description": data["description"], "genes": subset_genes}
    return subset
def get_gmt(file_path):
    gene_sets = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            name = parts[0]
            description = parts[1]
            genes = parts[2:]
            gene_sets[name] = {"description": description, "genes": genes}
    return gene_sets

def get_pathway_W(mat_gene,gmt_file):

    gSet = get_gmt(gmt_file)

    mat_gene_num_columns = mat_gene.shape[1]
    print(" ", mat_gene_num_columns)
    mat_gene_num_rows = mat_gene.shape[0]
    print("", mat_gene_num_rows)
    # 使用 np.apply_along_axis 计算 cells_rankings
    cells_rankings = np.apply_along_axis(lambda x: np.argsort(np.argsort(-x)), axis=1, arr=mat_gene.values)
    # 将结果转换为 DataFrame，并设置行名和列名
    df_cells_rankings = pd.DataFrame(cells_rankings, columns=mat_gene.columns, index=mat_gene.index)

    print(df_cells_rankings)
    print("------------------------------------------------------")
    print(df_cells_rankings.shape)
    df_cells_rankings_num_columns = len(df_cells_rankings.columns)
    print("Number of columns in df_cells_rankings DataFrame: ", df_cells_rankings_num_columns)
    df_cells_rankings_num_rows = len(df_cells_rankings)
    print("Number of rows in df_cells_rankings DataFrame: ", df_cells_rankings_num_rows)
    print("------------------------------------------------------")
    df_cells_rankings = df_cells_rankings.T
    mat_gene = mat_gene.T
    print(mat_gene)
    gSet = subsetGeneSets(gSet, mat_gene.index)
    gSet = pd.DataFrame.from_dict(gSet, orient='index')
    print(gSet)
    if gSet.size == 0:
        print("empty_way")
        auc_matrix = pd.DataFrame(np.zeros((len(gSet), mat_gene_num_rows)), columns=mat_gene.columns)
        return auc_matrix

    auc_matrix = pd.DataFrame(np.zeros((len(gSet), mat_gene_num_rows)), columns=mat_gene.columns)
    print(auc_matrix)
    # auc_matrix.index = gSet['description']
    print('-------------------------------------------------------')
    gSet_description = gSet['description'].reset_index(drop=True)
    # print(gSet_description)
    auc_matrix.index = gSet_description.tolist()
    print(auc_matrix)
    print('------------------------------------------------------')

    # auc_matrix.to_csv('auc_matrix.csv', index=True)
    gSet_num_columns = len(gSet.columns)
    print("Number of columns in gSet DataFrame: ", gSet_num_columns)
    gSet_num_rows = len(gSet)
    print("Number of rows in gSet DataFrame: ", gSet_num_rows)
    print(gSet['description'])


    print(mat_gene_num_rows)


    print(gSet_num_rows)

    # mat_gene_num_rows-3
    # gSet_num_rows
    #

    for i in range(gSet_num_rows):
        row = gSet.iloc[i]
        genes = row['genes']
        print(genes)
        description = row['description']
        for j in range(mat_gene_num_rows):
            df_cells_rankings_rows = df_cells_rankings.iloc[:, j]
            df_cells_rankings_column = df_cells_rankings_rows.loc[df_cells_rankings.index.isin(genes)]

            df_cells_rankings_column_array = df_cells_rankings_column.values

            df_cells_rankings_column_array = np.divide(df_cells_rankings_column_array, mat_gene_num_columns)

            df_cells_rankings_column_array = 1 - df_cells_rankings_column_array

            if np.size(df_cells_rankings_column_array) == 0:
                auc = 0

                auc_matrix.iat[j, i] = auc
            else:

                sorted_array = np.sort(df_cells_rankings_column_array)

                cumulative_sum = np.cumsum(sorted_array)

                normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]

                auc = np.trapz(sorted_array, normalized_cumulative_sum)

                auc_matrix.iat[i, j] = auc


    print(mat_gene_num_rows)
    print(gSet_num_rows)
    auc_matrix = auc_matrix.drop(auc_matrix.index[(auc_matrix == 0).all(axis=1)])
    return auc_matrix
