import argparse, numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

def main(dataset_id):
    spatial_matrix = np.loadtxt(f"data/{dataset_id}_loc.tsv", delimiter='\t')
    cell_expressions = pd.read_csv(f"data/{dataset_id}.tsv", sep='\t')
    he_image = np.array(Image.open(f"data/{dataset_id}.png").convert("RGB"))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    patch_size = 224
    half_size = patch_size // 2

    cell_ids_spatial = set(np.unique(spatial_matrix)) - {0}
    cell_ids_expr = set(cell_expressions["Unnamed: 0"])
    valid_cell_ids = sorted(cell_ids_spatial & cell_ids_expr)
    cell_expression_map = dict(zip(cell_expressions["Unnamed: 0"], cell_expressions.drop(columns=["Unnamed: 0"]).values))

    paired_patches, paired_expressions, used_cell_ids = [], [], []
    for cell_id in tqdm(valid_cell_ids):
        coords = np.argwhere(spatial_matrix == cell_id)
        if len(coords) == 0 or cell_id not in cell_expression_map: continue

        center_x = int(np.mean(coords[:, 0]))
        center_y = int(np.mean(coords[:, 1]))
        x_start = max(center_x - half_size, 0)
        y_start = max(center_y - half_size, 0)
        patch = he_image[x_start:x_start + patch_size, y_start:y_start + patch_size]
        if patch.shape != (patch_size, patch_size, 3): continue

        paired_patches.append(transform(Image.fromarray(patch)))
        paired_expressions.append(torch.tensor(cell_expression_map[cell_id], dtype=torch.float32))
        used_cell_ids.append(cell_id)

    torch.save(torch.stack(paired_patches), f"processed/{dataset_id}_tiles.pt")
    torch.save(torch.stack(paired_expressions), f"processed/{dataset_id}_vectors.pt")
    np.save(f"processed/{dataset_id}_ids.npy", np.array(used_cell_ids))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True)
    args = parser.parse_args()
    main(args.dataset_id)
