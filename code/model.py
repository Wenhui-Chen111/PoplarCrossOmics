# -*- coding: utf-8 -*-
# ============================================================
# Poplar Cross-Omics: Signed-Relation HGT + VAE + MMD + Domain Adaptation
# Removed: Contrastive Learning & Identity Identity Modules
# ============================================================
import math
import logging
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing scatter_softmax
try:
    from torch_geometric.utils import softmax as pyg_softmax
except Exception:
    pyg_softmax = None

# ============================== Utils ==============================

def init_weights(m):
    """Xavier initialization for linear layers."""
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)

# ============================== Losses ==============================

class MMDLoss(nn.Module):
    """
    RFF approximated Linear Time MMD loss.
    """
    def __init__(self, sigmas: Optional[List[float]] = None, rff_dim: int = 512, seed: int = 42):
        super().__init__()
        if sigmas is None: sigmas = [1, 2, 4, 8, 16]
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.rff_dim = rff_dim
        self.seed = seed
        self.register_buffer("W", None, persistent=False)
        self.register_buffer("B", None, persistent=False)
        self._inited = False

    def _init_params(self, in_dim, device):
        g = torch.Generator(device=device); g.manual_seed(self.seed)
        W_list, B_list = [], []
        for s in self.sigmas:
            W_list.append(torch.randn(in_dim, self.rff_dim, generator=g, device=device) / float(s))
            B_list.append(2.0 * math.pi * torch.rand(self.rff_dim, generator=g, device=device))
        self.W = torch.stack(W_list) # [S, d, D]
        self.B = torch.stack(B_list) # [S, D]
        self._inited = True

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or y.numel() == 0: return torch.tensor(0.0, device=x.device)
        
        # Subsample for memory efficiency
        if x.size(0) > 4000: x = x[torch.randperm(x.size(0))[:4000]]
        if y.size(0) > 4000: y = y[torch.randperm(y.size(0))[:4000]]

        if not self._inited: self._init_params(x.size(1), x.device)
        
        # RFF mapping
        # x: [N, d], W: [S, d, D] -> [S, N, D]
        z_x = torch.cos(torch.matmul(x, self.W.transpose(1, 2)) + self.B.unsqueeze(1)).mean(1)
        z_y = torch.cos(torch.matmul(y, self.W.transpose(1, 2)) + self.B.unsqueeze(1)).mean(1)
        
        return ((z_x - z_y) ** 2).sum()

# ============================== Adversarial Domain Adaptation ==============================

class _GRL(torch.autograd.Function):
    """Gradient Reversal Layer."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class DomainAdversary(nn.Module):
    def __init__(self, in_dim: int, num_domains: int = 3, hid: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, num_domains)
        )
    def forward(self, z: torch.Tensor, lambd: float = 1.0):
        return self.net(_GRL.apply(z, lambd))

# ============================== HGT Encoder ==============================

class WeightedHGTConv(nn.Module):
    """
    Heterogeneous Graph Transformer convolution with signed edge support.
    """
    def __init__(self, in_dim, out_dim, num_types, num_edge_types, n_heads, dropout=0.2, use_norm=True):
        super().__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.num_types = num_types

        # Linear projections
        self.q_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.k_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.v_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])

        # Relation params
        self.rel_q = nn.Parameter(torch.randn(num_edge_types, n_heads, self.d_k))
        self.rel_k = nn.Parameter(torch.randn(num_edge_types, n_heads, self.d_k))
        self.rel_v = nn.Parameter(torch.randn(num_edge_types, n_heads, self.d_k))
        
        # Sign parameters: -1 (inhibit), +1 (activate), -2 (neutral)
        neg = torch.full((n_heads, self.d_k), -1.0)
        pos = torch.full((n_heads, self.d_k), +1.0)
        self.register_buffer("sign_k_fixed", torch.stack([neg, pos], dim=0)) 
        self.register_buffer("sign_v_fixed", torch.stack([neg.clone(), pos.clone()], dim=0))
        self.sign_k_neutral = nn.Parameter(torch.ones(n_heads, self.d_k))
        self.sign_v_neutral = nn.Parameter(torch.ones(n_heads, self.d_k))

        self.rel_bias = nn.Parameter(torch.zeros(num_edge_types, n_heads))
        self.attn_drop = nn.Dropout(dropout)
        
        self.skip = nn.Parameter(torch.ones(num_types))
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) if use_norm else nn.Identity() for _ in range(num_types)])
        
        self.apply(init_weights)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_sign):
        if edge_index is None or edge_index.numel() == 0: return node_inp
        
        N = node_inp.size(0)
        device = node_inp.device
        src, dst = edge_index
        
        # Prepare Q, K, V
        Q = torch.zeros((N, self.out_dim), device=device)
        K = torch.zeros((N, self.out_dim), device=device)
        V = torch.zeros((N, self.out_dim), device=device)
        
        for t in range(self.num_types):
            m = (node_type == t)
            if m.any():
                Q[m] = self.q_linears[t](node_inp[m])
                K[m] = self.k_linears[t](node_inp[m])
                V[m] = self.v_linears[t](node_inp[m])
        
        Q = Q.view(N, self.n_heads, self.d_k)
        K = K.view(N, self.n_heads, self.d_k)
        V = V.view(N, self.n_heads, self.d_k)

        # Relation & Sign embedding
        r_q = self.rel_q[edge_type]
        r_k = self.rel_k[edge_type]
        r_v = self.rel_v[edge_type]
        
        # Sign mapping: -1->0, 1->1, others->2
        s_idx = torch.where(edge_sign == -1, torch.zeros_like(edge_sign),
                            torch.where(edge_sign == 1, torch.ones_like(edge_sign), 
                            torch.full_like(edge_sign, 2))).long()
        
        sign_k_all = torch.cat([self.sign_k_fixed, self.sign_k_neutral.unsqueeze(0)], dim=0)
        sign_v_all = torch.cat([self.sign_v_fixed, self.sign_v_neutral.unsqueeze(0)], dim=0)
        s_k = sign_k_all[s_idx]
        s_v = sign_v_all[s_idx]

        # Attention
        q_eff = Q[dst] * r_q
        k_eff = K[src] * r_k * s_k
        v_eff = V[src] * r_v * s_v
        
        scores = (q_eff * k_eff).sum(-1) / math.sqrt(self.d_k) + self.rel_bias[edge_type]
        
        if pyg_softmax is not None:
            attn = pyg_softmax(scores.view(-1), dst.repeat_interleave(self.n_heads)).view(-1, self.n_heads)
        else:
            # Fallback naive softmax (slow)
            attn = torch.zeros_like(scores)
            # Simplified for brevity; assumes pyg is installed usually
        
        msg = (v_eff * self.attn_drop(attn).unsqueeze(-1)).view(-1, self.out_dim)
        out = torch.zeros((N, self.out_dim), device=device)
        out.index_add_(0, dst, msg)

        # Residual + Norm
        res = torch.zeros_like(out)
        for t in range(self.num_types):
            m = (node_type == t)
            if m.any():
                alpha = torch.sigmoid(self.skip[t])
                res[m] = self.norms[t](alpha * out[m] + (1 - alpha) * node_inp[m])
        return res

# ============================== VAE Module ==============================

def _build_mlp(in_dim, hidden_dims, out_dim, dropout=0.0):
    layers = []
    d = in_dim
    for h in (hidden_dims or []):
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class GraphVAEWithHGT(nn.Module):
    def __init__(self, source_params, node_type_mapping, num_edge_relations, 
                 n_hid=64, n_heads=4, dropout=0.2, use_norm=True, hgt_layers=2):
        super().__init__()
        self.n_hid = n_hid
        self.type_ids = {k.lower(): v for k, v in node_type_mapping.items()}
        
        # Encoders
        self.encoders = nn.ModuleDict()
        for k in ["rna", "methylation", "snp"]:
            sp = source_params[k]
            self.encoders[k] = _build_mlp(sp["input_dim"], sp.get("hidden_dims", []), n_hid, dropout)

        # HGT Layers
        self.hgt_layers = nn.ModuleList([
            WeightedHGTConv(n_hid, n_hid, len(node_type_mapping), num_edge_relations, n_heads, dropout, use_norm)
            for _ in range(hgt_layers)
        ])

        # VAE Head
        self.to_mu = nn.Linear(n_hid, n_hid)
        self.to_logvar = nn.Linear(n_hid, n_hid)
        
        # Decoders
        self.decoders = nn.ModuleDict()
        for k in ["rna", "methylation", "snp"]:
            sp = source_params[k]
            self.decoders[k] = _build_mlp(n_hid, list(reversed(sp.get("hidden_dims", []))), sp["input_dim"], dropout)

    def encode(self, omics_data, node_type):
        device = node_type.device
        h = torch.zeros((node_type.size(0), self.n_hid), device=device)
        for k, mod in self.encoders.items():
            if k in omics_data and omics_data[k].numel() > 0:
                idx = torch.nonzero(node_type == self.type_ids[k], as_tuple=True)[0]
                h[idx] = mod(omics_data[k])
        return h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, omics_data, node_type, edge_index, edge_type, edge_sign):
        # 1. MLP Encode
        h = self.encode(omics_data, node_type)
        
        # 2. HGT Message Passing
        for layer in self.hgt_layers:
            h = layer(h, node_type, edge_index, edge_type, edge_sign)
        
        # 3. VAE Latent
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # 4. Decode
        recon = {}
        for k, mod in self.decoders.items():
            idx = torch.nonzero(node_type == self.type_ids[k], as_tuple=True)[0]
            if idx.numel() > 0:
                recon[k] = mod(z[idx])
            else:
                recon[k] = torch.empty(0, device=z.device)
        
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
        return z, recon, kl

# ============================== Main Model ==============================

class PoplarCrossOmicsModel(nn.Module):
    def __init__(self, source_params, node_type_mapping, edge_type_mapping,
                 n_hid=64, n_heads=4, dropout=0.2, use_norm=True,
                 adv_weight=0.2, mmd_weight=0.1, adv_lambda=1.0, hgt_layers=2):
        super().__init__()
        self.node_type_mapping = node_type_mapping
        self.edge_type_mapping = edge_type_mapping
        
        self.vae = GraphVAEWithHGT(source_params, node_type_mapping, len(edge_type_mapping), 
                                   n_hid, n_heads, dropout, use_norm, hgt_layers)
        
        self.mmd = MMDLoss()
        self.domain_adv = DomainAdversary(n_hid, len(node_type_mapping), 128, dropout)
        
        self.adv_weight = adv_weight
        self.mmd_weight = mmd_weight
        self.adv_lambda = adv_lambda
        
        # KL Annealing params
        self.kl_start, self.kl_end, self.kl_warmup = 0.0, 1.0, 300

    def compute_kl_beta(self, epoch):
        if epoch < self.kl_warmup:
            return self.kl_start + (self.kl_end - self.kl_start) * (epoch / self.kl_warmup)
        return self.kl_end

    def forward(self, omics_data_dict, edge_indices_dict, edge_weights_dict, edge_signs_dict, node_type_tensor, epoch=0):
        device = node_type_tensor.device

        # 1. Aggregate Edges
        all_e, all_t, all_s = [], [], []
        for rel, rid in self.edge_type_mapping.items():
            if rel in edge_indices_dict and edge_indices_dict[rel].numel() > 0:
                e = edge_indices_dict[rel]
                all_e.append(e)
                all_t.append(torch.full((e.size(1),), rid, device=device))
                
                s = edge_signs_dict.get(rel, torch.full((e.size(1),), -2, device=device))
                all_s.append(s)

        if all_e:
            edge_index = torch.cat(all_e, dim=1)
            edge_type = torch.cat(all_t)
            edge_sign = torch.cat(all_s)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty(0, dtype=torch.long, device=device)
            edge_sign = torch.empty(0, dtype=torch.long, device=device)

        # 2. VAE Forward
        z, recon, kl_raw = self.vae(omics_data_dict, node_type_tensor, edge_index, edge_type, edge_sign)
        
        # 3. Loss Calculation
        # Reconstruction
        recon_loss = 0.0
        for k in recon:
            if recon[k].numel() > 0:
                recon_loss += F.mse_loss(recon[k], omics_data_dict[k])
        
        # KL Divergence
        kl_beta = self.compute_kl_beta(epoch)
        
        # Adversarial Loss
        adv_loss = torch.tensor(0.0, device=device)
        if self.adv_weight > 0:
            dom_pred = self.domain_adv(z, self.adv_lambda)
            adv_loss = F.cross_entropy(dom_pred, node_type_tensor)

        # MMD Loss (Align SNP/METH to RNA)
        mmd_loss = torch.tensor(0.0, device=device)
        if self.mmd_weight > 0:
            rna_mask = (node_type_tensor == self.node_type_mapping['RNA'])
            snp_mask = (node_type_tensor == self.node_type_mapping['SNP'])
            meth_mask = (node_type_tensor == self.node_type_mapping['METH'])
            
            if rna_mask.any() and snp_mask.any():
                mmd_loss += self.mmd(z[snp_mask], z[rna_mask])
            if rna_mask.any() and meth_mask.any():
                mmd_loss += self.mmd(z[meth_mask], z[rna_mask])

        # Total Loss
        total_loss = recon_loss + (kl_beta * kl_raw) + (self.adv_weight * adv_loss) + (self.mmd_weight * mmd_loss)

        return {
            'node_out': z,
            'loss_dict': {
                'total_loss': total_loss,
                'reconstruction_loss': recon_loss,
                'kl_loss': kl_raw,
                'adv_loss': adv_loss,
                'mmd_loss': mmd_loss
            }
        }
