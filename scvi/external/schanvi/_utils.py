from anndata import AnnData
import scipy.sparse as sp_sparse
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import rich
from anndata import AnnData
from pandas.api.types import CategoricalDtype
from scvi.dataloaders._concat_dataloader import ConcatDataLoader
from scvi.data._utils import (
    _make_column_categorical
)
from scvi.data.fields import MuDataWrapper
from scvi.data.fields._obsm_field import JointObsField


import pytorch_lightning as pl

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._ann_dataloader import AnnDataLoader, BatchSampler
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._data_splitting import validate_data_split

#HIERARCHY :
"""
Dictionary for HLCA hierarchy.
"""
#hierarchy extracted by hand from the dataset  
hierarchy = {
    "Airway epithelium": ['Suprabasal',  'Goblet (nasal)', 'Ionocyte', 'Club (nasal)', 'Basal resting', 'Neuroendocrine', 'Multiciliated (non-nasal)',"Club (non-nasal)", "Transitional Club-AT2"],
    "Alveolar epithelium": [ "AT2", "AT1", 'AT2 proliferating', 'Ionocyte'],
    'Blood vessels' : ['EC arterial', 'EC venous systemic', "EC general capillary", "EC aerocyte capillary", "EC venous pulmonary"],
    'Fibroblast lineage' :['Alveolar fibroblasts', 'Myofibroblasts', 'Adventitial fibroblasts', 'Peribronchial fibroblasts', 'Pericytes'],
    'Lymphatic EC' :['Lymphatic EC mature'],
    'Lymphoid' :['CD8 T cells', 'CD4 T cells', 'NK cells', 'Plasma cells', 'B cells', 'T cells proliferating'],
    'Myeloid' :['Non-classical monocytes','Mast cells','Classical monocytes', 'Interstitial Mφ perivascular', 'Alveolar macrophages', 'Monocyte-derived Mφ', 'Alveolar Mφ proliferating', 'Migratory DCs', 'DC2', 'DC1', 'Plasmacytoid DCs'],
    'Smooth muscle' :['Smooth muscle', 'SM activated stress response', 'Fibromyocytes'] 
}

num_classes = [8, 44]

#helper function for testing purposes ; could be moved int he same file as generate_synthetic()
def _generate_synthetic_hierarchy(
    batch_size: int = 128,
    n_genes: int = 100,
    n_proteins: int = 100,
    n_batches: int = 2,
    n_labels_1: int = 2,
    n_labels_2: int = 10,
    n_labels_3: int = 11,
    sparse: bool = False,
) -> AnnData:
    """
    New method to generate test data with three-layer labels.
    """
    data = np.random.negative_binomial(5, 0.3, size=(batch_size * n_batches, n_genes))
    mask = np.random.binomial(n=1, p=0.7, size=(batch_size * n_batches, n_genes))
    data = data * mask  # We put the batch index first
    labels_1 = np.random.randint(0, n_labels_1, size=(batch_size * n_batches,))
    labels_2 = np.random.randint(0, n_labels_2, size=(batch_size * n_batches,))
    labels_3 = np.random.randint(0, n_labels_3, size=(batch_size * n_batches,))
    labels_1 = np.array(["label_%d" % i for i in labels_1])
    labels_2 = np.array(["label_%d" % i for i in labels_2])
    labels_3 = np.array(["label_%d" % i for i in labels_3])

    batch = []
    for i in range(n_batches):
        batch += [f"batch_{i}"] * batch_size

    if sparse:
        data = sp_sparse.csr_matrix(data)
    adata = AnnData(data)
    adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["labels_1"] = pd.Categorical(labels_1)
    adata.obs["labels_2"] = pd.Categorical(labels_2)
    adata.obs["labels_3"] = pd.Categorical(labels_3)

    # Protein measurements
    p_data = np.random.negative_binomial(5, 0.3, size=(adata.shape[0], n_proteins))
    adata.obsm["protein_expression"] = p_data
    adata.uns["protein_names"] = np.arange(n_proteins).astype(str)

    return adata

#helper function for testing purposes ; could be moved int he same file as synthetic_iid()
def synthetic_iid_hierarchy(
    batch_size: Optional[int] = 200,
    n_genes: Optional[int] = 100,
    n_proteins: Optional[int] = 100,
    n_batches: Optional[int] = 2,
    n_labels_1: Optional[int] = 2,
    n_labels_2: Optional[int] = 10,
    n_labels_3: Optional[int] = 11,
    sparse: bool = False,
) -> AnnData:
    """Synthetic dataset with ZINB distributed RNA and NB distributed protein, with three-layer annotation. 
    This dataset is just for testing purposed and not meant for modeling or research.
    Each value is independently and identically distributed.
    Parameters
    ----------
    batch_size
        Number of cells per batch
    n_genes
        Number of genes
    n_proteins
        Number of proteins
    n_batches
        Number of batches
    n_labels_1
        Number of cell types of layer 1 
    n_labels_2
        Number of cell types of layer 2
    sparse
        Whether to use a sparse matrix
    Returns
    -------
    AnnData with batch info (``.obs['batch']``), label info (``.obs['labels']``),
    protein expression (``.obsm["protein_expression"]``) and
    protein names (``.obs['protein_names']``)
    Examples
    --------
    >>> import scvi
    >>> adata = scvi.data.synthetic_iid()
    """
     
    return _generate_synthetic_hierarchy(
        batch_size=batch_size,
        n_genes=n_genes,
        n_proteins=n_proteins,
        n_batches=n_batches,
        n_labels_1=n_labels_1,
        n_labels_2 = n_labels_2,
        n_labels_3 = n_labels_3,
        sparse=sparse,
    )

#Class creating a new Obsm field for partially annotated layers of labels 
class LabelsWithUnlabeledJointObsField(JointObsField):
    """
    An AnnDataField for a collection of partially observed layers of labels .obs fields in the AnnData data structure.

    Creates an .obsm field compiling the given .obs fields. The model will reference the compiled
    data as a whole.

    Parameters
    ----------
    registry_key
        Key to register field under in data registry.
    obs_keys
        Sequence of keys to combine to form the obsm field.
    unlabeled_categories 
        Sequence of keys to account for unlabeled cells in every layer 
    """

    # don't know what that is
    MAPPINGS_KEY = "mappings"
    FIELD_KEYS_KEY = "field_keys"
    N_CATS_PER_KEY = "n_cats_per_key"
    UNLABELED_CATEGORIES = "unlabeled_categories"

    def __init__(self, registry_key: str, obs_keys: Optional[List[str]], unlabeled_categories: Optional[List[str]]) -> None:
        super().__init__(registry_key, obs_keys)
        self.count_stat_key = f"n_{self.registry_key}"
        self.unlabeled_categories = unlabeled_categories 


    def _default_mappings_dict(self) -> dict:
        return {
            self.MAPPINGS_KEY: dict(),
            self.FIELD_KEYS_KEY: [],
            self.N_CATS_PER_KEY: [],
            self.UNLABELED_CATEGORIES : [] 
        }

    def _make_obsm_categorical(
        self, adata: AnnData, category_dict: Optional[Dict[str, List[str]]] = None
    ) -> dict:
        if self.obs_keys != adata.obsm[self.attr_key].columns.tolist():
            raise ValueError(
                "Original .obs keys do not match the columns in the generated .obsm field."
            )

        categories = dict()
        obsm_df = adata.obsm[self.attr_key]
        for key in self.obs_keys: #for each layer 

            categorical_dtype = (
                CategoricalDtype(categories=category_dict[key])
                if category_dict is not None
                else None
            )
            mapping = _make_column_categorical(
                obsm_df, key, key, categorical_dtype=categorical_dtype
            )
            categories[key] = mapping

        store_cats = categories if category_dict is None else category_dict

        mappings_dict = self._default_mappings_dict()
        mappings_dict[self.MAPPINGS_KEY] = store_cats
        mappings_dict[self.FIELD_KEYS_KEY] = self.obs_keys
        mappings_dict[self.UNLABELED_CATEGORIES] = self.unlabeled_categories

        for k in self.obs_keys:
            mappings_dict[self.N_CATS_PER_KEY].append(len(store_cats[k]))
        return mappings_dict

    #is remap_unlabeled_to_final_category necessary here ? 

    def register_field(self, adata: AnnData) -> dict:
        super().register_field(adata)
        self._combine_obs_fields(adata)
        return self._make_obsm_categorical(adata)

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: AnnData,
        extend_categories: bool = False,
        **kwargs,
    ) -> dict:
        super().transfer_field(state_registry, adata_target, **kwargs)

        if self.is_empty:
            return

        source_cat_dict = state_registry[self.MAPPINGS_KEY].copy()
        if extend_categories:
            for key, mapping in source_cat_dict.items():
                for c in np.unique(adata_target.obs[key]):
                    if c not in mapping:
                        mapping = np.concatenate([mapping, [c]])
                source_cat_dict[key] = mapping

        self.validate_field(adata_target)
        self._combine_obs_fields(adata_target)
        return self._make_obsm_categorical(adata_target, category_dict=source_cat_dict)

    def get_summary_stats(self, _state_registry: dict) -> dict:
        n_obs_keys = len(self.obs_keys)

        return {
            self.count_stat_key: n_obs_keys,
        }

    def view_state_registry(self, state_registry: dict) -> Optional[rich.table.Table]:
        if self.is_empty:
            return None

        t = rich.table.Table(title=f"{self.registry_key} State Registry")
        t.add_column(
            "Source Location",
            justify="center",
            style="dodger_blue1",
            no_wrap=True,
            overflow="fold",
        )
        t.add_column(
            "Categories", justify="center", style="green", no_wrap=True, overflow="fold"
        )
        t.add_column(
            "scvi-tools Encoding",
            justify="center",
            style="dark_violet",
            no_wrap=True,
            overflow="fold",
        )
        for key, mappings in state_registry[self.MAPPINGS_KEY].items():
            for i, mapping in enumerate(mappings):
                if i == 0:
                    t.add_row("adata.obs['{}']".format(key), str(mapping), str(i))
                else:
                    t.add_row("", str(mapping), str(i))
            t.add_row("", "")
        return t


MuDataCategoricalJointObsField = MuDataWrapper(LabelsWithUnlabeledJointObsField)

class HierarchicalSemiSupervisedDataSplitter(pl.LightningDataModule):
    """
    Creates data loaders ``train_set``, ``validation_set``, ``test_set`` for scHANVI.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.
    The ratio between labeled and unlabeled data in adata will be preserved
    in the train/test/val sets.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    n_samples_per_label
        Number of subsamples for each label class to sample per epoch
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.external.SCHANVI.setup_anndata(adata, labels=["labels_0", "labels_1"], unknown_categories=["unknown, "unknown"])
    >>> adata_manager = scvi.external.SCHANVI(adata).adata_manager
    >>> unknown_label = 'label_0'
    >>> splitter = SemiSupervisedDataSplitter(adata_manager)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        n_samples_per_label: Optional[int] = None,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.data_loader_kwargs = kwargs
        self.n_samples_per_label = n_samples_per_label

        labels_state_registry = adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        )
        self.unlabeled_categories = labels_state_registry.unlabeled_categories

        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name, 
            adata_manager.data_registry.labels.attr_key,
        )
        self._label_mapping = labels_state_registry.mappings
        self._unlabeled_indices = np.where(labels.iloc[:,-1]==self.unlabeled_categories[-1])[0]
        self._labeled_indices = np.where(labels.iloc[:,-1]!=self.unlabeled_categories[-1])[0]
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu


    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_labeled_idx = len(self._labeled_indices)
        n_unlabeled_idx = len(self._unlabeled_indices)

        if n_labeled_idx != 0:
            n_labeled_train, n_labeled_val = validate_data_split(
                n_labeled_idx, self.train_size, self.validation_size
            )
            rs = np.random.RandomState(seed=settings.seed)
            labeled_permutation = rs.choice(
                self._labeled_indices, len(self._labeled_indices), replace=False
            )
            labeled_idx_val = labeled_permutation[:n_labeled_val]
            labeled_idx_train = labeled_permutation[
                n_labeled_val : (n_labeled_val + n_labeled_train)
            ]
            labeled_idx_test = labeled_permutation[(n_labeled_val + n_labeled_train) :]
        else:
            labeled_idx_test = []
            labeled_idx_train = []
            labeled_idx_val = []

        if n_unlabeled_idx != 0:
            n_unlabeled_train, n_unlabeled_val = validate_data_split(
                n_unlabeled_idx, self.train_size, self.validation_size
            )
            rs = np.random.RandomState(seed=settings.seed)
            unlabeled_permutation = rs.choice(
                self._unlabeled_indices, len(self._unlabeled_indices)
            )
            unlabeled_idx_val = unlabeled_permutation[:n_unlabeled_val]
            unlabeled_idx_train = unlabeled_permutation[
                n_unlabeled_val : (n_unlabeled_val + n_unlabeled_train)
            ]
            unlabeled_idx_test = unlabeled_permutation[
                (n_unlabeled_val + n_unlabeled_train) :
            ]
        else:
            unlabeled_idx_train = []
            unlabeled_idx_val = []
            unlabeled_idx_test = []

        indices_train = np.concatenate((labeled_idx_train, unlabeled_idx_train))
        indices_val = np.concatenate((labeled_idx_val, unlabeled_idx_val))
        indices_test = np.concatenate((labeled_idx_test, unlabeled_idx_test))
        self.train_idx = indices_train.astype(int)
        self.val_idx = indices_val.astype(int)
        self.test_idx = indices_test.astype(int)

        gpus = parse_use_gpu_arg(self.use_gpu, return_device=False)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )

        if len(self._labeled_indices) != 0:
            self.data_loader_class = HierarchicalSemiSupervisedDataLoader
            dl_kwargs = {
                "n_samples_per_label": self.n_samples_per_label,
            }
        else:
            self.data_loader_class = AnnDataLoader
            dl_kwargs = {}

        self.data_loader_kwargs.update(dl_kwargs)

    def train_dataloader(self):
        return self.data_loader_class(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=3,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=3,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=3,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

class HierarchicalSemiSupervisedDataLoader(ConcatDataLoader):
    """
    DataLoader that supports semisupervised training with multi-layer annotation.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    n_samples_per_label
        Number of subsamples for each label class to sample per epoch. By default, there
        is no label subsampling.
    indices
        The indices of the observations in the adata to load
    shuffle
        Whether the data should be shuffled
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (`adata_manager.data_registry`)
        and value equal to desired numpy loading type (later made into torch tensor).
        If `None`, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        n_samples_per_label: Optional[int] = None,
        indices: Optional[List[int]] = None,
        shuffle: bool = False,
        batch_size: int = 128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ):
        adata = adata_manager.adata
        if indices is None:
            indices = np.arange(adata.n_obs)

        self.indices = np.asarray(indices)

        if len(self.indices) == 0:
            return None

        self.n_samples_per_label = n_samples_per_label

        labels_state_registry = adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        )
        self.unlabeled_categories = labels_state_registry.unlabeled_categories

        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name, 
            adata_manager.data_registry.labels.attr_key,
        )
        # save a nested list of the indices per labeled category for the finest level 
        self.labeled_locs = []
        labels_fine = labels.iloc[:,-1].to_numpy()
        labels_fine.ravel()
        for label in np.unique(labels_fine):
            if label != labels_state_registry.unlabeled_categories[-1]:
                label_loc_idx = np.where(labels_fine[indices] == label)[0]
                label_loc = self.indices[label_loc_idx]
                self.labeled_locs.append(label_loc)
        labelled_idx = self.subsample_labels()

        super().__init__(
            adata_manager=adata_manager,
            indices_list=[self.indices, labelled_idx],
            shuffle=shuffle,
            batch_size=batch_size,
            data_and_attributes=data_and_attributes,
            drop_last=drop_last,
            **data_loader_kwargs,
        )

    def resample_labels(self):
        """Resamples the labeled data."""
        labelled_idx = self.subsample_labels()
        # self.dataloaders[0] iterates over full_indices
        # self.dataloaders[1] iterates over the labelled_indices
        # change the indices of the labelled set
        self.dataloaders[1].indices = labelled_idx

    def subsample_labels(self):
        """Subsamples each label class by taking up to n_samples_per_label samples per class."""
        if self.n_samples_per_label is None:
            return np.concatenate(self.labeled_locs)

        sample_idx = []
        for loc in self.labeled_locs:
            if len(loc) < self.n_samples_per_label:
                sample_idx.append(loc)
            else:
                label_subset = np.random.choice(
                    loc, self.n_samples_per_label, replace=False
                )
                sample_idx.append(label_subset)
        sample_idx = np.concatenate(sample_idx)
        return sample_idx
