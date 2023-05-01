import numpy as np 
"""
Dictionary for HLCA hierarchy.
"""
#hierarchy extracted by hand from the dataset and the cell ontology
#Basal resting? Migratory DCs? Pericytes ? Neuroendocrine ? Ionocyte?

hierarchy = {
    "Airway epithelium": ['Suprabasal', "EC general capillary", "EC aerocyte capillary",  'Goblet (nasal)', 'Ionocyte', 'Club (nasal)'],
    "Alveolar epithelium": ['Multiciliated (non-nasal)', "Club (non-nasal)", "AT2", "EC venous pulmonary", "Transitional Club-AT2", "AT1", 'AT2 proliferating', 'Neuroendocrine', 'Ionocyte'],
    'Blood vessels' : ['EC arterial', 'EC venous systemic', 'Pericytes'],
    'Fibroblast lineage' :['Alveolar fibroblasts', 'Myofibroblasts', 'Adventitial fibroblasts', 'Fibromyocytes', 'Peribronchial fibroblasts'],
    'Lymphatic EC' :['Lymphatic EC mature', 'Basal resting'],
    'Lymphoid' :['CD8 T cells', 'CD4 T cells', 'NK cells', 'Plasma cells', 'B cells','DC2','DC1','Plasmacytoid DCs', 'T cells proliferating', 'Migratory DCs'],
    'Fibroblast lineage' :['Alveolar fibroblasts', 'Myofibroblasts', 'Adventitial fibroblasts', 'Fibromyocytes', 'Peribronchial fibroblasts'],
    'Myeloid' :['Non-classical monocytes','Mast cells','Classical monocytes', 'Interstitial Mφ perivascular', 'Alveolar macrophages', 'Monocyte-derived Mφ', 'Alveolar Mφ proliferating'],
    'Smooth muscle' :['Smooth muscle', 'SM activated stress response'] 
}

num_classes = [8, 44]