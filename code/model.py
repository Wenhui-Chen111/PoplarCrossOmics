# -*- coding: utf-8 -*-
# 移除额外hgt模块
# ============================================================
# Poplar Cross-Omics: Signed-Relation HGT + VAE + MMD + Dir-Contrast + Calibration
# ============================================================
import math
import logging
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # if available, we use torch_geometric's softmax for grouped target-node softmax
    from torch_geometric.utils import softmax as pyg_softmax
except Exception:
    pyg_softmax = None


# ------------------------- utils -------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


class MMDLoss(nn.Module):
    """
    RFF 近似的线性时间 MMD（多带宽 + 可选对称分离梯度 + 可选子采样）
    """
    def __init__(self, 
                 sigmas: Optional[List[float]] = None,
                 rff_dim: int = 512,
                 kernel_weight: float = 0.8,
                 random_seed: int = 42):
        super().__init__()
        if sigmas is None:
            sigmas = [1, 2, 4, 8, 16]
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32, requires_grad=False)
        self.sigma_weights = nn.Parameter(torch.ones_like(self.sigmas) / len(self.sigmas), requires_grad=True)
        self.rff_dim = int(rff_dim)
        self.kernel_weight = float(kernel_weight)
        self.seed = int(random_seed)

        # 运行时内存控制开关（按需改 True/False）
        self.symmetric_detach = True   # 建议开：两次单侧回传，显著省显存
        self.max_samples = None        # 例如 8000；None 表示不用子采样

        # 惰性初始化 RFF 参数（每个 σ 一套）
        self.register_buffer("W", None, persistent=False)   # [S, d, D]
        self.register_buffer("B", None, persistent=False)   # [S, D]
        self.register_buffer("rff_scale", None, persistent=False)
        self._inited = False
        self._in_dim = None

    def _maybe_init(self, in_dim: int, device, dtype):
        if self._inited and self._in_dim == in_dim:
            # 保持 dtype/device 一致
            self.W = self.W.to(device=device, dtype=dtype)
            self.B = self.B.to(device=device, dtype=dtype)
            self.rff_scale = self.rff_scale.to(device=device, dtype=dtype)
            return

        g = torch.Generator(device=device); g.manual_seed(self.seed)
        S, D = self.sigmas.numel(), self.rff_dim
        W_list, B_list = [], []
        for s in self.sigmas.tolist():
            W_s = torch.randn(in_dim, D, generator=g, device=device, dtype=dtype) / float(s)
            b_s = 2.0 * math.pi * torch.rand(D, generator=g, device=device, dtype=dtype)
            W_list.append(W_s); B_list.append(b_s)
        self.W = nn.Parameter(torch.stack(W_list, dim=0), requires_grad=False)  # [S,d,D]
        self.B = nn.Parameter(torch.stack(B_list, dim=0), requires_grad=False)  # [S,D]
        self.rff_scale = nn.Parameter(torch.tensor((2.0 / float(D)) ** 0.5, device=device, dtype=dtype),
                                      requires_grad=False)
        self._inited, self._in_dim = True, in_dim

    @staticmethod
    def _maybe_subsample(Z: torch.Tensor, k: Optional[int]) -> torch.Tensor:
        if (k is None) or (Z.size(0) <= k):
            return Z
        idx = torch.randperm(Z.size(0), device=Z.device)[:k]
        return Z.index_select(0, idx)

    def _phi_mean(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [N,d]; W: [S,d,D]; B: [S,D] -> [S,N,D] -> mean->[S,D]
        Z1 = Z.unsqueeze(0)
        proj = torch.matmul(Z1, self.W) + self.B.unsqueeze(1)  # [S,N,D]
        phi = self.rff_scale * torch.cos(proj)                  # [S,N,D]
        return phi.mean(dim=1)                                  # [S,D]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or y.numel() == 0:
            return torch.tensor(0.0, device=x.device if x.numel() else y.device)

        # 可选：对子样本做 MMD，稳内存
        x = self._maybe_subsample(x, self.max_samples)
        y = self._maybe_subsample(y, self.max_samples)

        self._maybe_init(x.size(1), x.device, x.dtype)
        w = F.softmax(self.sigma_weights, dim=0).to(device=x.device, dtype=x.dtype)  # [S]

        mx = self._phi_mean(x)   # [S,D]
        my = self._phi_mean(y)   # [S,D]

        if self.symmetric_detach:
            # ½(||mx - sg(my)||^2 + ||sg(mx) - my||^2)
            d1 = (mx - my.detach()); d2 = (mx.detach() - my)
            per_sigma = 0.5 * ((d1 * d1).sum(dim=1) + (d2 * d2).sum(dim=1))  # [S]
        else:
            diff = mx - my
            per_sigma = (diff * diff).sum(dim=1)

        mmd_core = (w * per_sigma).sum()
        mmd = torch.clamp(self.kernel_weight * mmd_core, 0.0, 1e3)
        return mmd,mmd_core
    
# ==============================
# 2) GRL + 域判别器（对抗式对齐）
# ==============================
class _GRL(torch.autograd.Function):
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

# ==============================
# 1) WeightedHGTConv（三态符号，支持 -2 无方向；E==0 短路）
# ==============================
class WeightedHGTConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_edge_types, n_heads, dropout=0.2, use_norm=True):
        super().__init__()
        assert out_dim % n_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_edge_types = num_edge_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads

        # type-specific projections
        self.q_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.k_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.v_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])

        # relation gates [R, H, d_k]
        self.rel_q = nn.Parameter(torch.randn(num_edge_types, n_heads, self.d_k))
        self.rel_k = nn.Parameter(torch.randn(num_edge_types, n_heads, self.d_k))
        self.rel_v = nn.Parameter(torch.randn(num_edge_types, n_heads, self.d_k))
        nn.init.xavier_uniform_(self.rel_q); nn.init.xavier_uniform_(self.rel_k); nn.init.xavier_uniform_(self.rel_v)

        # sign gates: idx 0->-1, 1->+1, 2->-2(undirected/neutral)
        # self.sign_k = nn.Parameter(torch.ones(3, n_heads, self.d_k))
        # self.sign_v = nn.Parameter(torch.ones(3, n_heads, self.d_k))
        # with torch.no_grad():
        #     self.sign_k[0].fill_(-1.0); self.sign_v[0].fill_(-1.0)  # inhibition
        #     self.sign_k[1].fill_(+1.0); self.sign_v[1].fill_(+1.0)  # activation
        #     self.sign_k[2].fill_(+1.0); self.sign_v[2].fill_(+1.0)  # neutral
        # # 固定三套门控（防止不稳定），如需可学习中性缩放，可放开 index=2 的 requires_grad
        # self.sign_k.requires_grad_(False)
        # self.sign_v.requires_grad_(False)
        neg = torch.full((n_heads, self.d_k), -1.0)
        pos = torch.full((n_heads, self.d_k), +1.0)
        self.register_buffer("sign_k_fixed", torch.stack([neg, pos], dim=0))  # (2, H, d_k)
        self.register_buffer("sign_v_fixed", torch.stack([neg.clone(), pos.clone()], dim=0))

        # 可学习：未知/中性(-2) —— 初始化为 +1（即“默认不翻转”），训练中可自行调整
        self.sign_k_neutral = nn.Parameter(torch.ones(n_heads, self.d_k))     # (H, d_k)
        self.sign_v_neutral = nn.Parameter(torch.ones(n_heads, self.d_k))

        self.rel_bias = nn.Parameter(torch.zeros(num_edge_types, n_heads))
        self.dist_alpha = nn.Parameter(torch.tensor(0.0))
        self.dist_tau   = nn.Parameter(torch.tensor(1e5))

        self.attn_drop = nn.Dropout(dropout)
        self.msg_drop  = nn.Dropout(dropout)

        self.skip = nn.Parameter(torch.ones(num_types))
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) if use_norm else nn.Identity() for _ in range(num_types)])

        # init proj
        for ls in (self.q_linears, self.k_linears, self.v_linears):
            for lin in ls:
                nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self,
                node_inp: torch.Tensor,          # [N, in_dim]
                node_type: torch.Tensor,         # [N]
                edge_index: Optional[torch.Tensor],  # [2,E]
                edge_type: Optional[torch.Tensor],   # [E]
                edge_weights: Optional[torch.Tensor] = None,  # [E] (不用来兜底 sign 了)
                edge_sign: Optional[torch.Tensor] = None,     # [E] in {-1, +1, -2}
                edge_distance: Optional[torch.Tensor] = None, # [E]
                logger: Optional[logging.Logger] = None,
                epoch: int = 0,
                layer_idx: int = 0) -> torch.Tensor:

        N = node_inp.size(0)
        device = node_inp.device

        # E==0 或无边：短路返回
        if edge_index is None or edge_index.numel() == 0:
            return node_inp

        src = edge_index[0].to(device); dst = edge_index[1].to(device)
        E = src.numel()
        edge_type = edge_type.to(device)

        # 不再兜底从权重推断；未提供则全部视作 -2（无方向）
        if edge_sign is None:
            edge_sign = torch.full((E,), -2, dtype=torch.long, device=device)
        else:
            edge_sign = edge_sign.to(device)
            # 只接受 {-2, -1, +1}
            edge_sign = torch.where(edge_sign < -1, torch.full_like(edge_sign, -2),
                                    torch.where(edge_sign == 0, torch.full_like(edge_sign, -2),
                                                edge_sign.clamp(-1, 1)))

        # Q/K/V by type
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

        rel_q = self.rel_q[edge_type]  # [E,H,d]
        rel_k = self.rel_k[edge_type]
        rel_v = self.rel_v[edge_type]

        # sign idx: -1->0, +1->1, -2->2
        sign_idx = torch.where(edge_sign == -1, torch.zeros_like(edge_sign),
                               torch.where(edge_sign ==  1, torch.ones_like(edge_sign),
                                           torch.full_like(edge_sign, 2)))
        # s_k = self.sign_k[sign_idx]  # [E,H,d]
        # s_v = self.sign_v[sign_idx]
        sign_k_all = torch.cat([self.sign_k_fixed, self.sign_k_neutral.unsqueeze(0)], dim=0)  # (3,H,d_k)
        sign_v_all = torch.cat([self.sign_v_fixed, self.sign_v_neutral.unsqueeze(0)], dim=0)
        s_k = sign_k_all[sign_idx]
        s_v = sign_v_all[sign_idx]

        q_eff = Q[dst] * rel_q
        k_eff = K[src] * rel_k * s_k
        v_eff = V[src] * rel_v * s_v

        scores = (q_eff * k_eff).sum(-1) / math.sqrt(self.d_k)  # [E,H]
        scores = scores + self.rel_bias[edge_type]
        if edge_distance is not None:
            phi = self.dist_alpha * torch.exp(-edge_distance.to(device) / (self.dist_tau + 1e-9))
            scores = scores + phi.unsqueeze(-1)

        # softmax over incoming edges per dst per head
        if pyg_softmax is not None:
            attn = pyg_softmax(scores.view(-1), dst.repeat_interleave(self.n_heads)).view(E, self.n_heads)
        else:
            attn = torch.empty_like(scores)
            perm = torch.argsort(dst)
            d_sorted, s_sorted = dst[perm], scores[perm]
            cuts = torch.where(d_sorted[1:] != d_sorted[:-1])[0] + 1
            starts = torch.cat([torch.tensor([0], device=device), cuts])
            ends   = torch.cat([cuts, torch.tensor([E], device=device)])
            for s, e in zip(starts.tolist(), ends.tolist()):
                seg = s_sorted[s:e]
                m = seg.max(dim=0, keepdim=True).values
                es = torch.exp(seg - m); Z = es.sum(0, keepdim=True)
                attn[perm[s:e]] = es / (Z + 1e-9)

        attn = self.attn_drop(attn)
        msg = (v_eff * attn.unsqueeze(-1)).view(E, self.out_dim)

        out = torch.zeros((N, self.out_dim), device=device)
        out.index_add_(0, dst, msg)

        res = torch.zeros_like(out)
        for t in range(self.num_types):
            m = (node_type == t)
            if m.any():
                alpha = torch.sigmoid(self.skip[t])
                res[m] = self.norms[t](alpha * out[m] + (1 - alpha) * node_inp[m])
        return res


def _build_mlp(in_dim, hidden_dims, out_dim, dropout=0.0, last_act=False):
    layers = []
    d = in_dim
    for h in (hidden_dims or []):
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    if last_act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class GraphVAEWithHGT(nn.Module):
    """
    编码：
      - 为 RNA / METH / SNP 各自建一个 MLP 编码器，把各自特征 → 统一维度 n_hid
      - 按 node_type_tensor 把编码后的向量放回到全体节点 [N, n_hid] 的对应位置
      - 可选两层 HGT（没有边则短路返回 h0）
    VAE：
      - mu/logvar 从 HGT 输出得到；reparameterize 得 z
      - 每个组学各自一个解码器：z[该类型] → 重构回原特征维度
    返回：
      latent_dict = {"z": z}
      recon_dict  = {"rna": ..., "methylation": ..., "snp": ...}
      stats       = {"z": z, "kl": kl}
    """
    def __init__(self,
                 source_params: dict,
                 node_type_mapping: dict,
                 num_edge_relations: int,
                 n_hid: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.2,
                 use_norm: bool = True,
                 hgt_layers: int = 2  # 新增：HGT层数配置（默认2层，和原代码一致）
                 ):
        super().__init__()
        self.n_hid = n_hid
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_norm = use_norm
        self.hgt_layers = hgt_layers  # 保存层数，供forward使用
        # 节点类型ID（原逻辑不变）
        self.node_type_mapping = dict(node_type_mapping)
        self.type_ids = {
            "rna":  self.node_type_mapping["RNA"],
            "methylation": self.node_type_mapping["METH"],
            "snp":  self.node_type_mapping["SNP"],
        }

        # ===================== 组学编码器（原逻辑不变） =====================
        self.encoders = nn.ModuleDict()
        for omics_key in ["rna", "methylation", "snp"]:
            sp = source_params[omics_key]
            in_dim = int(sp["input_dim"])
            hid = list(sp.get("hidden_dims", []))
            self.encoders[omics_key] = _build_mlp(
                in_dim=in_dim,
                hidden_dims=hid,
                out_dim=n_hid,  # 组学编码器输出维度= n_hid（和HGT输入对齐）
                dropout=dropout,
                last_act=False
            )

        # ===================== 可配置HGT层（核心优化） =====================
        # 用ModuleList存储HGT层，层数由 hgt_layers 控制
        self.hgt_layers_list = nn.ModuleList()
        for _ in range(self.hgt_layers):
            self.hgt_layers_list.append(
                WeightedHGTConv(
                    in_dim=n_hid, out_dim=n_hid,  # HGT层输入输出维度一致（均为n_hid）
                    num_types=len(self.node_type_mapping),
                    num_edge_types=num_edge_relations,
                    n_heads=n_heads,
                    dropout=dropout,
                    use_norm=use_norm
                )
            )

        # ===================== VAE头（原逻辑不变） =====================
        self.to_mu     = nn.Linear(n_hid, n_hid)
        self.to_logvar = nn.Linear(n_hid, n_hid)
        self.decoders = nn.ModuleDict()
        for omics_key in ["rna", "methylation", "snp"]:
            sp = source_params[omics_key]
            out_dim = int(sp["input_dim"])
            hid_rev = list(reversed(sp.get("hidden_dims", [])))
            self.decoders[omics_key] = _build_mlp(
                in_dim=n_hid,
                hidden_dims=hid_rev,
                out_dim=out_dim,
                dropout=dropout,
                last_act=False
            )

    # -------------------- 关键：把各组学编码到全体节点 --------------------
    @torch.no_grad()
    def _indices_of_type(self, node_type_tensor: torch.Tensor, type_id: int) -> torch.Tensor:
        # 返回该类型在全体节点中的位置索引（1D LongTensor）
        return torch.nonzero(node_type_tensor == type_id, as_tuple=False).squeeze(1)

    def encode_omics(self, omics_data: dict, node_type_tensor: torch.Tensor) -> torch.Tensor:
        """
        omics_data: {'rna': [Nr, Din_rna], 'methylation': [Nm, Din_met], 'snp': [Ns, Din_snp]}
        node_type_tensor: [N] (含三种类型的 ID，需与 self.node_type_mapping 对齐)
        返回 h0: [N, n_hid]
        """
        device = node_type_tensor.device
        N = node_type_tensor.size(0)
        h0 = torch.zeros((N, self.n_hid), device=device)

        # RNA
        if "rna" in omics_data and omics_data["rna"].numel() > 0:
            idx = self._indices_of_type(node_type_tensor, self.type_ids["rna"])
            x = omics_data["rna"]
            assert x.size(0) == idx.numel(), f"RNA 行数({x.size(0)})必须等于 RNA 节点数({idx.numel()})"
            h0[idx] = self.encoders["rna"](x)

        # METH
        if "methylation" in omics_data and omics_data["methylation"].numel() > 0:
            idx = self._indices_of_type(node_type_tensor, self.type_ids["methylation"])
            x = omics_data["methylation"]
            assert x.size(0) == idx.numel(), f"METH 行数({x.size(0)})必须等于 METH 节点数({idx.numel()})"
            h0[idx] = self.encoders["methylation"](x)

        # SNP
        if "snp" in omics_data and omics_data["snp"].numel() > 0:
            idx = self._indices_of_type(node_type_tensor, self.type_ids["snp"])
            x = omics_data["snp"]
            assert x.size(0) == idx.numel(), f"SNP 行数({x.size(0)})必须等于 SNP 节点数({idx.numel()})"
            h0[idx] = self.encoders["snp"](x)

        return h0

    # -------------------- VAE 常规部件 --------------------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_omics(self, z: torch.Tensor, node_type_tensor: torch.Tensor) -> dict:
        """
        将全体节点的 z 分别送入各自解码器，重构回原特征维度。
        返回：{'rna': [Nr, Din_rna], 'methylation': [Nm, Din_met], 'snp': [Ns, Din_snp]}
        """
        out = {}
        # RNA
        idx = self._indices_of_type(node_type_tensor, self.type_ids["rna"])
        out["rna"] = self.decoders["rna"](z[idx]) if idx.numel() > 0 else torch.empty(0, self.decoders["rna"][-1].out_features, device=z.device)
        # METH
        idx = self._indices_of_type(node_type_tensor, self.type_ids["methylation"])
        out["methylation"] = self.decoders["methylation"](z[idx]) if idx.numel() > 0 else torch.empty(0, self.decoders["methylation"][-1].out_features, device=z.device)
        # SNP
        idx = self._indices_of_type(node_type_tensor, self.type_ids["snp"])
        out["snp"] = self.decoders["snp"](z[idx]) if idx.numel() > 0 else torch.empty(0, self.decoders["snp"][-1].out_features, device=z.device)
        return out

    # -------------------- 主前向 --------------------
    def forward(self,
                omics_data: dict,
                node_type_tensor: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                edge_weights: torch.Tensor,
                edge_sign: torch.Tensor,
                edge_distance: torch.Tensor,
                kl_beta: float,
                logger=None,
                epoch: int = 0):
        # 1) 组学编码得到h0（原逻辑不变）
        h0 = self.encode_omics(omics_data, node_type_tensor)  # [N, n_hid]

        # 2) 多HGT层循环传播（层数由 self.hgt_layers 控制）
        h = h0  # 初始输入
        if edge_index is not None and edge_index.numel() > 0:  # 有边才走HGT，无边短路
            for layer_idx, hgt_layer in enumerate(self.hgt_layers_list, start=1):
                h = hgt_layer(
                    node_inp=h,
                    node_type=node_type_tensor,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    edge_weights=edge_weights,
                    edge_sign=edge_sign,
                    edge_distance=edge_distance,
                    logger=logger,
                    epoch=epoch,
                    layer_idx=layer_idx  # 传递层索引，便于日志调试
                )

        # 3) VAE（原逻辑不变）
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = self.reparameterize(mu, logvar)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

        # 4) 解码（原逻辑不变）
        recon = self.decode_omics(z, node_type_tensor)
        latent_dict = {"z": z}
        stats = {"z": z, "kl": kl}
        return latent_dict, recon, stats


# ==============================
# 4) DirectionalContrastiveHead（三态；-2 边不计入损失）
# ==============================
class DirectionalContrastiveHead(nn.Module):
    def __init__(self, proj_dim: int, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.src_proj = nn.Linear(proj_dim, proj_dim, bias=False)
        self.dst_proj = nn.Linear(proj_dim, proj_dim, bias=False)

    def forward(self, node_emb: torch.Tensor,
                pos_pairs: torch.Tensor,   # [3,P]: src, dst, sign_idx (0:-1, 1:+1, 2:-2)
                neg_pairs: torch.Tensor    # [2,P*K]
                ) -> torch.Tensor:
        if pos_pairs.numel() == 0:
            return torch.tensor(0.0, device=node_emb.device)

        z = F.normalize(node_emb, p=2, dim=-1)
        u = F.normalize(self.src_proj(z[pos_pairs[0]]), p=2, dim=-1)
        v = F.normalize(self.dst_proj(z[pos_pairs[1]]), p=2, dim=-1)
        sim_pos = (u * v).sum(-1)  # [P]

        sign_idx = pos_pairs[2].long()
        # -1: 取反；+1: 保持；-2: 中性（不参与损失）
        mult = torch.ones_like(sim_pos)
        mult = torch.where(sign_idx == 0, -torch.ones_like(sim_pos), mult)
        sim_pos = sim_pos * mult
        use_mask = (sign_idx != 2)  # 仅计算 ±1 的损失

        # 如全部为中性，返回 0
        if not use_mask.any():
            return torch.tensor(0.0, device=node_emb.device)

        u_neg = F.normalize(self.src_proj(z[neg_pairs[0]]), p=2, dim=-1)
        v_neg = F.normalize(self.dst_proj(z[neg_pairs[1]]), p=2, dim=-1)
        sim_neg = (u_neg * v_neg).sum(-1).view(u.size(0), -1)

        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temperature
        labels = torch.zeros(u.size(0), dtype=torch.long, device=node_emb.device)

        # 只对 ±1 的样本累积交叉熵；中性样本权重为 0
        loss_all = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss_all * use_mask.float()).sum() / (use_mask.float().sum() + 1e-9)
        return loss


# ==============================
# 3) 负采样（逐样本 sign；-2 则不约束相似度）
# ==============================
def _edge2key(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # 64-bit pairing; 假设节点数 < 2^32
    return (u.to(torch.int64) << 32) | v.to(torch.int64)

class CrossOmicsNegativeSampler:
    def __init__(self, source_pool: torch.Tensor, target_pool: torch.Tensor,
                 existing_edges_tensor: Optional[torch.Tensor] = None, neg_k: int = 4, device: Optional[torch.device] = None):
        self.device = device or source_pool.device
        self.source_pool = source_pool.to(self.device)
        self.target_pool = target_pool.to(self.device)
        self.neg_k = neg_k
        if existing_edges_tensor is not None and existing_edges_tensor.numel() > 0:
            ek = _edge2key(existing_edges_tensor[:, 0], existing_edges_tensor[:, 1])
            self.exist_keys_sorted, _ = torch.sort(ek.to(self.device))
        else:
            self.exist_keys_sorted = None

    def sample_negatives(self, pos_pairs: torch.Tensor, node_emb: torch.Tensor,
                         sign_per_pos: torch.Tensor, neg_per_pos: Optional[int] = None) -> torch.Tensor:
        """
        pos_pairs: [2, P]
        sign_per_pos: [P] in {-1, +1, -2}
        return: [2, P*K]
        """
        if pos_pairs.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        K = neg_per_pos or self.neg_k
        P = pos_pairs.size(1)
        need = P * K

        # 同类型随机负样本
        src_rep = pos_pairs[0].repeat_interleave(K)               # [P*K]
        rand_dst = self.target_pool[torch.randint(0, len(self.target_pool), (need,), device=self.device)]
        cand = torch.stack([src_rep, rand_dst], dim=1)            # [P*K, 2]

        # 去除已存在的正边
        if self.exist_keys_sorted is not None:
            keys = _edge2key(cand[:, 0], cand[:, 1])
            pos = torch.searchsorted(self.exist_keys_sorted, keys).clamp(max=self.exist_keys_sorted.numel()-1)
            dup = (self.exist_keys_sorted.numel() > 0) & (self.exist_keys_sorted[pos] == keys)
            cand = cand[~dup]
            # 不足则补齐
            while cand.size(0) < need:
                deficit = need - cand.size(0)
                add_dst = self.target_pool[torch.randint(0, len(self.target_pool), (deficit,), device=self.device)]
                add = torch.stack([src_rep[:deficit], add_dst], dim=1)
                add_keys = _edge2key(add[:, 0], add[:, 1])
                pos2 = torch.searchsorted(self.exist_keys_sorted, add_keys).clamp(max=self.exist_keys_sorted.numel()-1)
                cand = torch.cat([cand, add[self.exist_keys_sorted[pos2] != add_keys]], dim=0)
            cand = cand[:need]

        # 逐样本相似度规则
        z = F.normalize(node_emb, p=2, dim=-1)
        neg_src = z[cand[:, 0]]
        neg_dst = z[cand[:, 1]]
        neg_sim = (neg_src * neg_dst).sum(-1)                     # [P*K]

        pos_sim = (z[pos_pairs[0]] * z[pos_pairs[1]]).sum(-1)     # [P]
        pos_sim_rep = pos_sim.repeat_interleave(K)                # [P*K]
        sign_rep = sign_per_pos.repeat_interleave(K)              # [P*K]

        # keep：+1 要比正样本更不相似；-1 要比正样本更相似；-2 保留全部
        keep_pos = (sign_rep == 1)  & (neg_sim < pos_sim_rep)
        keep_neg = (sign_rep == -1) & (neg_sim > pos_sim_rep)
        keep_neu = (sign_rep == -2)
        keep = keep_pos | keep_neg | keep_neu
        kept = cand[keep]

        # 数量不足时补齐（按各自规则）
        if kept.size(0) < need:
            deficit = need - kept.size(0)
            if (sign_rep == 1).any():
                idx = torch.argsort(neg_sim)[:deficit]
            elif (sign_rep == -1).any():
                idx = torch.argsort(neg_sim, descending=True)[:deficit]
            else:
                idx = torch.arange(min(deficit, cand.size(0)), device=self.device)
            kept = torch.cat([kept, cand[idx]], dim=0)

        return kept[:need].t().contiguous()



# ==============================
# 6) 主模型（图可为空；对抗对齐；三态对比学习）
# ==============================
class PoplarCrossOmicsModel(nn.Module):
    def __init__(self,
                 source_params: Dict[str, Dict],
                 node_type_mapping: Dict[str, int],
                 edge_type_mapping: Dict[str, int],
                 n_hid: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.2,
                 use_norm: bool = True,
                 contrastive_weight: float = 0.0,
                 neg_per_pos: int = 4,
                 adv_weight: float = 0.2,
                 mmd_weight: float = 0.1,
                 adv_lambda: float = 1.0,
                 hgt_layers: int = 2  # 新增：HGT层数（传递给GraphVAEWithHGT）
                 ):
        super().__init__()
        self.node_type_mapping = node_type_mapping
        self.edge_type_mapping = edge_type_mapping
        self.num_node_types = len(node_type_mapping)
        self.num_edge_relations = len(edge_type_mapping)
        self.identity_weight = 0      
        self.identity_tau = 0.2
        self.identity_abs_cos = True    

        # ===================== 传递hgt_layers给GraphVAEWithHGT =====================
        self.graph_vae = GraphVAEWithHGT(
            source_params=source_params,
            node_type_mapping=self.node_type_mapping,
            num_edge_relations=self.num_edge_relations,
            n_hid=n_hid, 
            n_heads=n_heads,
            dropout=dropout, 
            use_norm=use_norm,
            hgt_layers=hgt_layers  # 把层数参数传进去
        )

        # 后续损失组件、超参初始化（原逻辑不变）
        self.mmd = MMDLoss([1,2,4,8,16])   
        self.contrastive_head = DirectionalContrastiveHead(proj_dim=n_hid, temperature=0.2)
        self.domain_adv = DomainAdversary(in_dim=n_hid, num_domains=self.num_node_types, hid=128, dropout=dropout)
        self.contrastive_weight = contrastive_weight
        self.neg_per_pos = neg_per_pos
        self.adv_weight = adv_weight
        self.mmd_weight = mmd_weight
        self.adv_lambda = adv_lambda
        self.kl_beta_start, self.kl_beta_end, self.kl_warmup_epochs = 0.0, 1.0, 10
        self.apply(init_weights)
    
    def _same_gene_pairs(self, all_nodes_ordered, node_type_tensor, src_type="RNA", dst_type="METH"):
        # 基于 all_nodes_ordered 生成“同名”成对索引（src_idx, dst_idx），只要两模态都存在的基因
        id_map = {i: gid for i, (t, gid) in enumerate(all_nodes_ordered)}
        src_id = self.node_type_mapping[src_type]; dst_id = self.node_type_mapping[dst_type]
        src_idx = [i for i in range(len(node_type_tensor)) if node_type_tensor[i].item()==src_id]
        dst_idx = [i for i in range(len(node_type_tensor)) if node_type_tensor[i].item()==dst_id]
        gid_src = {id_map[i]: i for i in src_idx}; gid_dst = {id_map[i]: i for i in dst_idx}
        common = [g for g in gid_src.keys() if g in gid_dst]
        if not common: return None
        src_list = [gid_src[g] for g in common]
        dst_list = [gid_dst[g] for g in common]
        return torch.tensor(src_list, dtype=torch.long, device=node_type_tensor.device), \
            torch.tensor(dst_list, dtype=torch.long, device=node_type_tensor.device)

    def _identity_infoNCE(self, emb, idx_q, idx_ref, tau=0.2, abs_cos=True, sample=2048):
        # 取子样本做 in-batch InfoNCE（内存友好）
        if idx_q.numel()==0 or idx_ref.numel()==0: return emb.new_tensor(0.0)
        n = min(sample, idx_q.numel(), idx_ref.numel())
        perm = torch.randperm(n, device=emb.device)
        q = F.normalize(emb[idx_q[perm]], p=2, dim=-1)
        r = F.normalize(emb[idx_ref[perm]], p=2, dim=-1)

        sim = q @ r.t()
        if abs_cos: sim = sim.abs()
        logits = sim / tau
        labels = torch.arange(n, device=emb.device)
        return F.cross_entropy(logits, labels)

    def compute_kl_beta(self, epoch: int) -> float:
        if epoch < self.kl_warmup_epochs:
            r = epoch / max(1, self.kl_warmup_epochs)
            return self.kl_beta_start + (self.kl_beta_end - self.kl_beta_start) * r
        return self.kl_beta_end

     # -------------------- 【核心修改：forward函数（删除额外HGT层 + 删除输出投影）】 --------------------
    def forward(self,
                omics_data_dict: Dict[str, torch.Tensor],
                edge_indices_dict: Dict[str, torch.Tensor],
                edge_weights_dict: Dict[str, torch.Tensor],
                edge_signs_dict: Dict[str, torch.Tensor],
                node_type_tensor: torch.Tensor,  # [N]
                all_nodes_ordered: Optional[List[Tuple[str, str]]] = None,
                epoch: int = 0,
                logger: Optional[logging.Logger] = None,
                return_post_hoc: bool = False):
        device = node_type_tensor.device

        # -------------------- 1. 合并多关系边（保留原逻辑，不修改） --------------------
        all_edges, all_types, all_weights, all_signs = [], [], [], []
        for rel_name, rel_id in self.edge_type_mapping.items():
            eidx = edge_indices_dict.get(rel_name, None)
            if eidx is None or eidx.numel() == 0:
                continue
            E_r = eidx.size(1)
            all_edges.append(eidx.to(device))
            all_types.append(torch.full((E_r,), rel_id, dtype=torch.long, device=device))
            w = edge_weights_dict.get(rel_name, torch.ones(E_r, device=device))
            all_weights.append(w.to(device))
            if rel_name in edge_signs_dict and edge_signs_dict[rel_name] is not None:
                s = edge_signs_dict[rel_name].to(device)
            else:
                s = torch.full((E_r,), -2, dtype=torch.long, device=device)
            s = torch.where(s < -1, torch.full_like(s, -2),
                            torch.where(s == 0, torch.full_like(s, -2), s.clamp(-1, 1)))
            all_signs.append(s)
        if len(all_edges) == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_type  = torch.empty(0, dtype=torch.long, device=device)
            edge_w     = torch.empty(0, device=device)
            edge_s     = torch.empty(0, dtype=torch.long, device=device)
        else:
            edge_index = torch.cat(all_edges, dim=1)
            edge_type  = torch.cat(all_types, dim=0)
            edge_w     = torch.cat(all_weights, dim=0)
            edge_s     = torch.cat(all_signs,  dim=0)

        # -------------------- 2. VAE前向（保留原逻辑，不修改） --------------------
        kl_beta = self.compute_kl_beta(epoch)
        latent_dict, recon_dict, vae_stats = self.graph_vae(
            omics_data=omics_data_dict,
            node_type_tensor=node_type_tensor,
            edge_index=edge_index, edge_type=edge_type,
            edge_weights=edge_w, edge_sign=edge_s,
            edge_distance=None,
            kl_beta=kl_beta, logger=logger, epoch=epoch
        )

        # -------------------- 3. 损失计算（保留原逻辑：重构+KL+对抗+MMD+身份对齐+对比学习） --------------------
        # 3.1 重构损失 + KL损失
        recon_loss = 0.0
        recon_per_omics = {}
        for source in ["snp", "methylation", "rna"]:
            if source in recon_dict and source in omics_data_dict and omics_data_dict[source].numel() > 0:
                l = F.mse_loss(recon_dict[source], omics_data_dict[source])
                recon_per_omics[source] = l; recon_loss += l
            else:
                recon_per_omics[source] = torch.tensor(0.0, device=device)
        kl_loss = vae_stats["kl"]
        loss = recon_loss + kl_beta * kl_loss

        # 3.2 对抗式域对齐（接到VAE的z上，因额外HGT层已删）
        adv_loss = torch.tensor(0.0, device=device)
        current_emb = vae_stats["z"]  # 【关键：直接用VAE的z作为核心特征，无额外HGT】
        if self.adv_weight > 0.0 and self.domain_adv is not None:
            dom_logits = self.domain_adv(current_emb, lambd=self.adv_lambda)
            targets = node_type_tensor.to(dom_logits.device, non_blocking=True)
            adv_loss = F.cross_entropy(dom_logits, targets)
            loss = loss + self.adv_weight * adv_loss

        # 3.3 MMD损失（SNP-RNA + METH-RNA）
        z_snp = current_emb[(node_type_tensor == self.node_type_mapping["SNP"])]
        z_met = current_emb[(node_type_tensor == self.node_type_mapping["METH"])]
        z_rna = current_emb[(node_type_tensor == self.node_type_mapping["RNA"])]
        mmd_sr, _ = self.mmd(z_snp, z_rna)
        mmd_mr, _ = self.mmd(z_met, z_rna)

        mmd_loss = mmd_sr + mmd_mr
        loss = loss + self.mmd_weight* mmd_loss

        # 3.4 身份一致性损失（同名基因对齐）
        id_loss = torch.tensor(0.0, device=device)
        # if all_nodes_ordered is not None and self.identity_weight > 0.0:
        #     pair_rm = self._same_gene_pairs(all_nodes_ordered, node_type_tensor, "RNA", "METH")
        #     pair_rs = self._same_gene_pairs(all_nodes_ordered, node_type_tensor, "RNA", "SNP")
        #     if pair_rm is not None:
        #         l_rm = self._identity_infoNCE(current_emb, pair_rm[0], pair_rm[1],
        #                                     tau=self.identity_tau, abs_cos=self.identity_abs_cos)
        #         id_loss = id_loss + l_rm
        #     if pair_rs is not None:
        #         l_rs = self._identity_infoNCE(current_emb, pair_rs[0], pair_rs[1],
        #                                     tau=self.identity_tau, abs_cos=self.identity_abs_cos)
        #         id_loss = id_loss + l_rs
        loss = loss + self.identity_weight * id_loss

        # 3.5 对比学习损失
        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_monitor = {}
        if self.training and self.contrastive_weight > 0.0 and edge_index.numel() > 0:
            all_pos, all_neg = [], []
            id_to_type = {v: k for k, v in self.node_type_mapping.items()}
            for rel_name, eidx in edge_indices_dict.items():
                if eidx is None or eidx.numel() == 0:
                    continue
                sgn = edge_signs_dict.get(rel_name, None)
                if sgn is None or sgn.numel() == 0:
                    sgn = torch.full((eidx.size(1),), -2, dtype=torch.long, device=device)
                else:
                    sgn = torch.where(sgn < -1, torch.full_like(sgn, -2),
                                      torch.where(sgn == 0, torch.full_like(sgn, -2), sgn.clamp(-1, 1)))
                src0, dst0 = eidx[0, 0].item(), eidx[1, 0].item()
                src_type = id_to_type[node_type_tensor[src0].item()]
                dst_type = id_to_type[node_type_tensor[dst0].item()]
                src_pool = torch.nonzero(node_type_tensor == self.node_type_mapping[src_type], as_tuple=False).squeeze(1)
                dst_pool = torch.nonzero(node_type_tensor == self.node_type_mapping[dst_type], as_tuple=False).squeeze(1)
                if src_pool.numel() == 0 or dst_pool.numel() == 0:
                    continue
                sampler = CrossOmicsNegativeSampler(src_pool, dst_pool, existing_edges_tensor=eidx.t(),
                                                    neg_k=self.neg_per_pos, device=device)
                neg_pairs = sampler.sample_negatives(eidx, current_emb, sign_per_pos=sgn, neg_per_pos=self.neg_per_pos)
                if neg_pairs.numel() == 0:
                    continue
                sign_idx = torch.where(sgn == -1, torch.zeros_like(sgn),
                                       torch.where(sgn == 1, torch.ones_like(sgn),
                                                   torch.full_like(sgn, 2)))
                pos = torch.stack([eidx[0].to(device), eidx[1].to(device), sign_idx], dim=0)
                all_pos.append(pos); all_neg.append(neg_pairs)
                # 对比学习监控（保留原逻辑）
                z_unit = F.normalize(current_emb, p=2, dim=-1)
                pos_sim_raw = (z_unit[eidx[0]] * z_unit[eidx[1]]).sum(-1).mean().item()
                neg_sim_raw = (z_unit[neg_pairs[0]] * z_unit[neg_pairs[1]]).sum(-1).mean().item()
                contrastive_monitor[rel_name] = dict(pos_mean=round(pos_sim_raw,4), neg_mean=round(neg_sim_raw,4))
            if len(all_pos) > 0:
                pos_pairs = torch.cat(all_pos, dim=1)
                neg_pairs = torch.cat(all_neg, dim=1)
                contrastive_loss = self.contrastive_head(current_emb, pos_pairs, neg_pairs)
                loss = loss + self.contrastive_weight * contrastive_loss



        loss_dict = {
            "total_loss": loss,
            "reconstruction_loss": recon_loss,
            "recon_rna": recon_per_omics["rna"],
            "recon_methylation": recon_per_omics["methylation"],
            "recon_snp": recon_per_omics["snp"],
            "kl_loss": kl_loss,
            "mmd_loss": mmd_loss,
            "contrastive_loss": contrastive_loss,
            "adv_loss": adv_loss if self.adv_weight > 0 else torch.tensor(0.0, device=device),
            "id_loss": id_loss
        }
        outputs = {
            "loss_dict": loss_dict,
            "node_emb": current_emb.detach(),  # 中间特征：VAE的z（保留，与原逻辑兼容）
            "node_out": current_emb.detach(),  # 【关键修改】最终输出=VAE的z，无投影
            "contrastive_monitor": contrastive_monitor
        }
        if return_post_hoc:
            outputs["post_hoc_scores"] = self.compute_post_hoc_edge_scores(current_emb, edge_indices_dict)
        return outputs

    # -------------------- 保留原有post_hoc计算（不修改） --------------------
    @torch.no_grad()
    def compute_post_hoc_edge_scores(self, node_emb: torch.Tensor, edge_indices_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = F.normalize(node_emb, p=2, dim=-1)
        post = {}
        for rel_name, eidx in edge_indices_dict.items():
            if eidx is None or eidx.numel() == 0:
                post[rel_name] = torch.empty(0, device=node_emb.device); continue
            post[rel_name] = (z[eidx[0]] * z[eidx[1]]).sum(-1)
        return post