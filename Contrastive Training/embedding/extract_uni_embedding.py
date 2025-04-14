import argparse
import torch
import os
import timm

# Check for Hugging Face login
if not os.path.exists(os.path.expanduser("~/.huggingface/token")):
    raise RuntimeError("You must login to Hugging Face first")

# Extract image embeddings using pretrained UNI model
def main(dataset_id):
    image_patches = torch.load(f"processed/{dataset_id}_tiles.pt")

    model = timm.create_model(
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
        act_layer=torch.nn.SiLU,
    ).cuda().eval()

    image_patches = image_patches.cuda()
    with torch.no_grad():
        uni_features = model(image_patches)  # [N, 1536]

    torch.save(uni_features.cpu(), f"processed/{dataset_id}_uni_features.pt")
    print(f"Saved: processed/{dataset_id}_uni_features.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True)
    args = parser.parse_args()
    main(args.dataset_id)
