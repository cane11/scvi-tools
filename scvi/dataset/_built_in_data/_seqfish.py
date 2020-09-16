import logging
import anndata
import os
import zipfile
import pandas as pd
import numpy as np

from scvi.dataset._built_in_data._utils import _download
from scvi.dataset import setup_anndata

logger = logging.getLogger(__name__)


def _load_seqfishplus(
    save_path: str = "data/",
    tissue_region: str = "subventricular cortex",
    run_setup_anndata: bool = True,
) -> anndata.AnnData:

    if tissue_region == "subventricular cortex":
        file_prefix = "cortex_svz"
    elif tissue_region == "olfactory bulb":
        file_prefix = "ob"
    else:
        raise ValueError(
            '`tissue_type` must be "subventricular cortex" or "olfactory bulb", but got {}'.format(
                tissue_region
            )
        )

    save_path = os.path.abspath(save_path)
    url = "https://github.com/CaiGroup/seqFISH-PLUS/raw/master/sourcedata.zip"
    save_fn = "seqfishplus.zip"

    _download(url, save_path, save_fn)
    adata = _load_seqfishplus_data(
        os.path.join(save_path, save_fn), file_prefix, save_path, gene_by_cell=False
    )
    adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
    adata.obs["labels"] = np.zeros(adata.shape[0], dtype=np.int64)

    if run_setup_anndata:
        setup_anndata(adata, batch_key="batch", labels_key="labels")
    return adata


def _load_seqfishplus_data(
    path_to_file: str, file_prefix: str, save_path: str, gene_by_cell: bool = False
) -> anndata.AnnData:
    counts_filename = "sourcedata/{}_counts.csv".format(file_prefix)
    coordinates_filename = "sourcedata/{}_cellcentroids.csv".format(file_prefix)
    extract_location = os.path.join(save_path, "seqfishplus")
    if not os.path.exists(extract_location):
        os.makedirs(extract_location)
    with zipfile.ZipFile(path_to_file) as f:
        f.extract(counts_filename, path=extract_location)
        f.extract(coordinates_filename, path=extract_location)

    df_counts = pd.read_csv(os.path.join(extract_location, counts_filename))
    adata = anndata.AnnData(df_counts)
    adata.var_names = df_counts.columns
    df_coordinates = pd.read_csv(os.path.join(extract_location, coordinates_filename))

    adata.obs["X"] = df_coordinates["X"].values
    adata.obs["Y"] = df_coordinates["Y"].values
    adata.obs["cell_id"] = df_coordinates["Cell ID"].values
    adata.obs["field_of_view"] = df_coordinates["Field of View"].values

    return adata


def _load_seqfish(
    save_path: str = "data/", run_setup_anndata: bool = True
) -> anndata.AnnData:
    save_path = os.path.abspath(save_path)
    url = "https://www.cell.com/cms/attachment/2080562255/2072099886/mmc6.xlsx"
    save_fn = "SeqFISH.xlsx"
    _download(url, save_path, save_fn)
    adata = _load_seqfish_data(os.path.join(save_path, save_fn))
    adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
    adata.obs["labels"] = np.zeros(adata.shape[0], dtype=np.int64)
    if run_setup_anndata:
        setup_anndata(adata, batch_key="batch", labels_key="labels")
    return adata


def _load_seqfish_data(path_to_file: str) -> anndata.AnnData:
    logger.info("Loading seqfish dataset from {}".format(path_to_file))
    xl = pd.ExcelFile(path_to_file)
    counts = xl.parse("Hippocampus Counts")
    data = (
        counts.values[:, 1:].astype(int).T
    )  # transpose because counts is genes X cells
    gene_names = counts.values[:, 0].astype(str)
    adata = anndata.AnnData(pd.DataFrame(data=data, columns=gene_names))
    logger.info("Finished loading seqfish dataset")
    return adata
