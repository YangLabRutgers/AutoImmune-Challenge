import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.histo_encoder import HistoEncoder
import timm

class EmbeddingPairDataset(Dataset):
    def __init__(self, emb1, emb2):
        self.emb1 = emb1
        self.emb2 = emb2

    def __len__(self):
        return len(self.emb1)

    def __getitem__(self, idx):
        return {
            'img_emb': self.emb1[idx],
            'expr_emb': self.emb2[idx],
        }

def soft_contrastive_loss(emb1, emb2, temperature=0.1):
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    sim11 = emb1 @ emb1.T
    sim22 = emb2 @ emb2.T
    soft_targets = (sim11 + sim22) / 2.0
    logits = emb1 @ emb2.T / temperature
    log_probs_12 = F.log_softmax(logits, dim=1)
    log_probs_21 = F.log_softmax(logits.T, dim=1)
    soft_targets = F.softmax(soft_targets / temperature, dim=1)
    loss_12 = -torch.sum(soft_targets * log_probs_12, dim=1).mean()
    loss_21 = -torch.sum(soft_targets * log_probs_21, dim=1).mean()
    return (loss_12 + loss_21) / 2

def main(dataset_id):
    image_embeddings = torch.load(f"processed/{dataset_id}_uni_features.pt")
    expression_embeddings = torch.load(f"processed/{dataset_id}_neighbor_features.pt")
    assert image_embeddings.shape == expression_embeddings.shape

    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    expression_embeddings = F.normalize(expression_embeddings, p=2, dim=1)

    dataset = EmbeddingPairDataset(image_embeddings, expression_embeddings)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    projection = torch.nn.Sequential(
        torch.nn.Linear(1536, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128)
    ).cuda()

    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-4)
    best_loss = float('inf')
    wait = 0

    for epoch in range(200):
        projection.train()
        total_loss = 0
        for batch in loader:
            img_emb = projection(batch['img_emb'].cuda())
            expr_emb = projection(batch['expr_emb'].cuda())
            loss = soft_contrastive_loss(img_emb, expr_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(projection.state_dict(), f"processed/{dataset_id}_projection.pt")
            print(" Saved best projection head.")
        else:
            wait += 1
            if wait >= 15:
                print(" Early stopping triggered.")
                break

    # Final: Save full HistoEncoder
    uni_model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        embed_dim=1536,
        num_classes=0,
        no_embed_class=True,
        reg_tokens=8,
        dynamic_img_size=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU
    ).eval()

    histo_model = HistoEncoder(uni_model, projection)
    torch.save(histo_model, f"processed/{dataset_id}_histo_encoder_full.pt")
    print(f"Saved full model: processed/{dataset_id}_histo_encoder_full.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True)
    args = parser.parse_args()
    main(args.dataset_id)
