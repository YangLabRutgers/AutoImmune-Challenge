import os

DATASETS = ["DC5", "DC6", "DC7"]

for ds in DATASETS:
    os.system(f"python preprocessing/preprocess_pairing.py --dataset_id {ds}")
    os.system(f"python embedding/extract_uni_embedding.py --dataset_id {ds}")
    os.system(f"python embedding/extract_neighbor_embedding.py --dataset_id {ds}")
    os.system(f"python training/contrastive_train.py --dataset_id {ds}")
