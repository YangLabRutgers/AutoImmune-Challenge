import torch
import torch.nn as nn
import torch.nn.functional as F

class HistoEncoder(nn.Module):
    def __init__(self, uni_model, projection_head):
        super().__init__()
        self.uni = uni_model.eval()
        self.proj = projection_head

    def forward(self, image_patch):
        if image_patch.ndim == 3:
            image_patch = image_patch.unsqueeze(0)
        with torch.no_grad():
            feat = self.uni(image_patch)
        out = self.proj(feat)
        return F.normalize(out, dim=1)

    def save_model(self, path):
        torch.save(self, path)

    @staticmethod
    def load_model(path):
        model = torch.load(path)
        model.eval()
        return model
