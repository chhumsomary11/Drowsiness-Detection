import torch
import torch.nn as nn
import torchvision.models as models


class BranchEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return x


class MultiBranchDrowsinessModel(nn.Module):
    def __init__(self, branch_dim=128, num_classes=2):
        super().__init__()

        self.eyes_branch = BranchEncoder(out_dim=branch_dim)
        self.mouth_branch = BranchEncoder(out_dim=branch_dim)
        self.face_branch = BranchEncoder(out_dim=branch_dim)

        fusion_dim = branch_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, eyes, mouth, face):
        eyes_feat = self.eyes_branch(eyes)
        mouth_feat = self.mouth_branch(mouth)
        face_feat = self.face_branch(face)

        fused = torch.cat([eyes_feat, mouth_feat, face_feat], dim=1)
        out = self.classifier(fused)
        return out