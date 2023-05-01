import numpy as np

from scvi.external.schanvi._utils import synthetic_iid_hierarchy
from scvi.external import SCHANVI

def test_schanvi() : 
    adata = synthetic_iid_hierarchy()
    SCHANVI.setup_anndata(adata, labels=["labels_1","labels_2", "labels_3"], unlabeled_categories=["unknown","unknown", 'Unknown'], batch_key="batch")
    model = SCHANVI(adata)
    model.train(max_epochs=1, train_size=0.5)


