"""
SparseTemporalPIE v3 (reconstructed for evaluation)
=====================================================
Architecture:
  - EfficientPIE backbone
  - Pose: 68-d (34 static + 34 velocity) projected to 1280-d, fused into embeddings
  - Multi-frame cross-attention: Q=f_current, K/V=K context frames (K<=4)
  - Feedforward block: 1280 -> 512 -> 1280
  - Context MLP (late fusion): bbox_traj(12) + ctx_feats(5) -> 128-d
  - Classifier: 1408 -> 256 -> 2

Forward: (f_current, f_context, context_mask, pose_current, pose_context, bbox_traj, ctx_feats)
"""

import torch
import torch.nn as nn
from models.EfficientPIE import EfficientPIE


class SparseTemporalPIE_v3(nn.Module):
    def __init__(self, dropout=0.1, pose_dim=68, traj_dim=12, ctx_dim=5,
                 num_heads=8, ff_hidden=512):
        super().__init__()

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

        # Pose projection: 68-d (static + velocity) -> 1280, shared for current + context
        self.pose_proj = nn.Linear(pose_dim, embed_dim, bias=False)

        # Multi-token cross-attention (batch_first=False: seq, batch, embed)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                                batch_first=False)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Feedforward block
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # Context MLP: bbox trajectory + ctx feats -> 128
        self.ctx_proj = nn.Sequential(
            nn.Linear(traj_dim + ctx_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

        # Classifier: 1280 + 128 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, f_current, f_context, context_mask, pose_current, pose_context,
                bbox_traj, ctx_feats):
        """
        Args:
            f_current:     (B, 3, 300, 300)
            f_context:     (B, K, 3, 300, 300)
            context_mask:  (B, K) — 1.0=real frame, 0.0=padding
            pose_current:  (B, 68)
            pose_context:  (B, K, 68)
            bbox_traj:     (B, 12)
            ctx_feats:     (B, 5)
        Returns:
            logits: (B, 2)
        """
        B, K = f_context.shape[:2]

        # 1. Batch all frames through backbone
        all_frames = torch.cat([f_current.unsqueeze(1), f_context], dim=1)  # (B, K+1, 3, H, W)
        all_embs = self.backbone(all_frames.view(-1, 3, 300, 300)).view(B, K + 1, 1280)

        emb_current = all_embs[:, 0]    # (B, 1280)
        emb_context = all_embs[:, 1:]   # (B, K, 1280)

        # 2. Fuse pose
        emb_current = emb_current + self.pose_proj(pose_current)
        emb_context = emb_context + self.pose_proj(
            pose_context.view(B * K, 68)).view(B, K, 1280)

        # 3. Multi-token cross-attention (batch_first=False)
        q  = emb_current.unsqueeze(0)          # (1, B, 1280)
        kv = emb_context.permute(1, 0, 2)      # (K, B, 1280)
        pad_mask = (context_mask == 0)          # (B, K) True = ignore
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=pad_mask)
        attn_out = attn_out.squeeze(0)          # (B, 1280)

        enriched = self.attn_norm(emb_current + attn_out)
        enriched = self.ff_norm(enriched + self.ff(enriched))

        # 4. Context features (late fusion)
        ctx = self.ctx_proj(torch.cat([bbox_traj, ctx_feats], dim=-1))  # (B, 128)

        # 5. Classify
        return self.classifier(torch.cat([enriched, ctx], dim=-1))
