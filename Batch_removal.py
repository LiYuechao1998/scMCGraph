import numpy as np
import scanpy as sc
import pandas as pd
import io

def Batch_removal(Traindataname_truefeatures, Traindataname_truelabels, Testdataname_truefeatures, Testdataname_truelabels):
    Traindataname_truefeatures.obs['cell_type'] = Traindataname_truelabels['cell_type']
    sc.pp.filter_cells(Traindataname_truefeatures, min_genes=200)
    sc.pp.filter_genes(Traindataname_truefeatures, min_cells=3)
    sc.pp.normalize_total(Traindataname_truefeatures, target_sum=1e4)
    sc.pp.log1p(Traindataname_truefeatures)
    sc.pp.highly_variable_genes(Traindataname_truefeatures, n_top_genes=2000)

    Testdataname_truefeatures.obs['cell_type'] = Testdataname_truelabels['cell_type']
    sc.pp.filter_cells(Testdataname_truefeatures, min_genes=200)
    sc.pp.filter_genes(Testdataname_truefeatures, min_cells=3)
    sc.pp.normalize_total(Testdataname_truefeatures, target_sum=1e4)
    sc.pp.log1p(Testdataname_truefeatures)
    sc.pp.highly_variable_genes(Testdataname_truefeatures, n_top_genes=2000)

    common_highly_variable_genes = set(Traindataname_truefeatures.var_names[Traindataname_truefeatures.var['highly_variable']]).intersection(
        set(Testdataname_truefeatures.var_names[Testdataname_truefeatures.var['highly_variable']]))
    common_highly_variable_genes = list(common_highly_variable_genes)
    Traindataname_truefeatures = Traindataname_truefeatures[:, Traindataname_truefeatures.var_names.isin(common_highly_variable_genes)]
    Testdataname_truefeatures = Testdataname_truefeatures[:, Testdataname_truefeatures.var_names.isin(common_highly_variable_genes)]
    Traindataname_truefeatures_obs_df = Traindataname_truefeatures.obs[['cell_type']]

    Testdataname_truefeatures_obs_df = Testdataname_truefeatures.obs[['cell_type']]

    Traindataname_truefeatures_X_df = pd.DataFrame(Traindataname_truefeatures.X.toarray(), columns=Traindataname_truefeatures.var_names)

    Testdataname_truefeatures_X_df = pd.DataFrame(Testdataname_truefeatures.X.toarray(), columns=Testdataname_truefeatures.var_names)


    Traindataname_truefeatures_obs_names = pd.DataFrame(Traindataname_truefeatures.obs_names, columns=[''])
    Traindataname_fintruefeatures = pd.concat([Traindataname_truefeatures_obs_names, Traindataname_truefeatures_X_df], axis=1)
    Traindataname_fintruefeatures.set_index('', inplace=True)
    print(Traindataname_fintruefeatures)


    Testdataname_truefeatures_obs_names = pd.DataFrame(Testdataname_truefeatures.obs_names, columns=[''])
    Testdataname_fintruefeatures = pd.concat([Testdataname_truefeatures_obs_names, Testdataname_truefeatures_X_df], axis=1)
    Testdataname_fintruefeatures.set_index('', inplace=True)
    print(Testdataname_fintruefeatures)



    Traindataname_fintruefeatures_csv_str = Traindataname_fintruefeatures.to_csv(index_label='')

    Traindataname_fintruefeatures_df = pd.read_csv(io.StringIO(Traindataname_fintruefeatures_csv_str), header=0, index_col=0)

    print(Traindataname_fintruefeatures_df)

    Traindataname_fintruefeatures_df = Traindataname_fintruefeatures_df.T

    mat_gene = np.array(Traindataname_fintruefeatures_df)



    merged_data = sc.AnnData.concatenate(
        Traindataname_truefeatures,
        Testdataname_truefeatures,
        join="inner",
        batch_key="batch",
        batch_categories=["Traindataname_truefeatures", "Testdataname_truefeatures"]
    )

    sc.pp.pca(merged_data)

    print(merged_data)

    sc.external.pp.harmony_integrate(merged_data, "batch")
    print(merged_data)

    merged_data.obsm['X_harmony'] = merged_data.X

    merged_harmony_data = merged_data.obsm['X_pca_harmony']

    Traindataname_merged_harmony_data = merged_harmony_data[merged_data.obs["batch"] == "Traindataname_truefeatures", :]
    Testdataname_merged_harmony_data = merged_harmony_data[merged_data.obs["batch"] == "Testdataname_truefeatures", :]

    return Traindataname_fintruefeatures, Testdataname_fintruefeatures, Traindataname_merged_harmony_data, Testdataname_merged_harmony_data, Traindataname_truefeatures_obs_df, Testdataname_truefeatures_obs_df