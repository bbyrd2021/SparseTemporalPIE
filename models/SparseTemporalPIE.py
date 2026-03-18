"""
SparseTemporalPIE v4
====================
Architecture (simplified from v3):
  - EfficientPIE backbone (partially unfrozen) — single frame pass only
  - Pose projection: 34-d static → 1280-d, fused into backbone embedding
  - Context MLP: bbox trajectory (12-d) + ctx features (5-d) → 128-d (late fusion)
  - Classifier: 1280 + 128 = 1408 → 256 → 2

Removed from v3:
  - Cross-attention (K+1 backbone passes → 1 pass, cleaner IL gradient flow)
  - Feedforward transformer block
  - Pose velocity (34-d → dropped, static only)
  - Context frame images (f_context, context_mask, pose_context)

Rationale: cross-attention on 1-4 near-identical frame embeddings adds noise
without signal. Temporal information is encoded more directly via bbox_traj
(where the pedestrian is moving) and ctx_feats (speed, action, look).

Authors: Brandon Byrd, Abel Abebe Bzuayene -- xDI Lab, NC A&T State University
"""

import torch
import torch.nn as nn
from models.EfficientPIE import EfficientPIE


class SparseTemporalPIE(nn.Module):
    def __init__(self, dropout=0.1, pose_dim=34, traj_dim=12, ctx_dim=5):
        super().__init__()

        # -- Backbone (partially unfrozen during training) -----------------
        _base = EfficientPIE()
        self.backbone = nn.Sequential(
            _base.commonConv,
            _base.fm1,
            _base.fm2,
            _base.mb1,
            _base.mb2,
            _base.commonConv1,
            _base.avg_pool,
            _base.flatten,
            _base.dropout,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = 1280

        # -- Pose projection (34-d static) → 1280, fused into embedding ----
        self.pose_proj = nn.Linear(pose_dim, embed_dim, bias=False)

        # -- Context MLP: bbox trajectory + speed/action/look → 128 -------
        self.ctx_proj = nn.Sequential(
            nn.Linear(traj_dim + ctx_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

        # -- Classifier: 1280 + 128 = 1408 → 2 ----------------------------
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def encode(self, x):
        """Pass (B, 3, 300, 300) through backbone → (B, 1280)."""
        return self.backbone(x)

    def forward(self, f_current, pose_current, bbox_traj, ctx_feats):
        """
        Args:
            f_current:    (B, 3, 300, 300)
            pose_current: (B, 34)  — normalized static keypoints
            bbox_traj:    (B, 12) — trajectory statistics over [0, step]
            ctx_feats:    (B, 5)  — [speed, speed_mean, speed_valid, action, look]

        Returns:
            logits: (B, 2)
        """
        # 1. Backbone + pose fusion
        emb = self.encode(f_current) + self.pose_proj(pose_current)  # (B, 1280)

        # 2. Context features (late fusion)
        ctx = self.ctx_proj(torch.cat([bbox_traj, ctx_feats], dim=-1))  # (B, 128)

        # 3. Classify
        return self.classifier(torch.cat([emb, ctx], dim=-1))  # (B, 2)


def load_backbone_weights(model, weights_path, device='cuda:0'):
    """
    Load trained EfficientPIE weights into the SparseTemporalPIE backbone.
    """
    checkpoint = torch.load(weights_path, map_location=device)
    src = checkpoint.get('model', checkpoint)

    key_map = {
        'commonConv':  '0',
        'fm1':         '1',
        'fm2':         '2',
        'mb1':         '3',
        'mb2':         '4',
        'commonConv1': '5',
        'avg_pool':    '6',
        'flatten':     '7',
        'dropout':     '8',
    }

    target = {}
    for src_key, src_val in src.items():
        for name, idx in key_map.items():
            if src_key.startswith(name + '.'):
                new_key = 'backbone.' + idx + src_key[len(name):]
                target[new_key] = src_val
                break

    missing, unexpected = model.load_state_dict(target, strict=False)
    loaded = len(target) - len(unexpected)
    print(f"Backbone weights loaded: {loaded}/{len(src)} keys")
    if missing:
        print(f"  Missing (new layers, expected): {len(missing)} keys")
    return model
