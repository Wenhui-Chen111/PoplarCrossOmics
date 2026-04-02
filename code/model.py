import math
import logging
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import softmax as pyg_softmax
except Exception:
    pyg_softmax = None

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


class MMDLoss(nn.Module):
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

        self.symmetric_detach = True  
        self.max_samples = None      

        self.register_buffer("W", None, persistent=False)   # [S, d, D]
        self.register_buffer("B", None, persistent=False)   # [S, D]
        self.register_buffer("rff_scale", None, persistent=False)
        self._inited = False
        self._in_dim = None

    def _maybe_init(self, in_dim: int, device, dtype):
        if self._inited and self._in_dim == in_dim:

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


        neg = torch.full((n_heads, self.d_k), -1.0)
        pos = torch.full((n_heads, self.d_k), +1.0)
        self.register_buffer("sign_k_fixed", torch.stack([neg, pos], dim=0))  # (2, H, d_k)
        self.register_buffer("sign_v_fixed", torch.stack([neg.clone(), pos.clone()], dim=0))

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
                node_inp: torch.Tensor,        
                node_type: torch.Tensor,       
                edge_index: Optional[torch.Tensor],  
                edge_type: Optional[torch.Tensor],  
                edge_weights: Optional[torch.Tensor] = None, 
                edge_sign: Optional[torch.Tensor] = None,    
                edge_distance: Optional[torch.Tensor] = None )-> torch.Tensor:
      

        N = node_inp.size(0)
        device = node_inp.device

        if edge_index is None or edge_index.numel() == 0:
            return node_inp

        src = edge_index[0].to(device); dst = edge_index[1].to(device)
        E = src.numel()
        edge_type = edge_type.to(device)


        if edge_sign is None:
            edge_sign = torch.full((E,), -2, dtype=torch.long, device=device)
        else:
            edge_sign = edge_sign.to(device)

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


        sign_idx = torch.where(edge_sign == -1, torch.zeros_like(edge_sign),
                               torch.where(edge_sign ==  1, torch.ones_like(edge_sign),
                                           torch.full_like(edge_sign, 2)))

        sign_k_all = torch.cat([self.sign_k_fixed, self.sign_k_neutral.unsqueeze(0)], dim=0)  
        sign_v_all = torch.cat([self.sign_v_fixed, self.sign_v_neutral.unsqueeze(0)], dim=0)
        s_k = sign_k_all[sign_idx]
        s_v = sign_v_all[sign_idx]

        q_eff = Q[dst] * rel_q
        k_eff = K[src] * rel_k * s_k
        v_eff = V[src] * rel_v * s_v

        scores = (q_eff * k_eff).sum(-1) / math.sqrt(self.d_k)
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
    def __init__(self,
                 source_params: dict,
                 node_type_mapping: dict,
                 num_edge_relations: int,
                 n_hid: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.2,
                 use_norm: bool = True,
                 hgt_layers: int = 2  
                 ):
        super().__init__()
        self.n_hid = n_hid
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_norm = use_norm
        self.hgt_layers = hgt_layers  
        self.node_type_mapping = dict(node_type_mapping)
        self.type_ids = {
            "rna":  self.node_type_mapping["RNA"],
            "methylation": self.node_type_mapping["METH"],
            "snp":  self.node_type_mapping["SNP"],
        }


        self.encoders = nn.ModuleDict()
        for omics_key in ["rna", "methylation", "snp"]:
            sp = source_params[omics_key]
            in_dim = int(sp["input_dim"])
            hid = list(sp.get("hidden_dims", []))
            self.encoders[omics_key] = _build_mlp(
                in_dim=in_dim,
                hidden_dims=hid,
                out_dim=n_hid,  
                dropout=dropout,
                last_act=False
            )


        self.hgt_layers_list = nn.ModuleList()
        for _ in range(self.hgt_layers):
            self.hgt_layers_list.append(
                WeightedHGTConv(
                    in_dim=n_hid, out_dim=n_hid, 
                    num_types=len(self.node_type_mapping),
                    num_edge_types=num_edge_relations,
                    n_heads=n_heads,
                    dropout=dropout,
                    use_norm=use_norm
                )
            )


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

    @torch.no_grad()
    def _indices_of_type(self, node_type_tensor: torch.Tensor, type_id: int) -> torch.Tensor:
        return torch.nonzero(node_type_tensor == type_id, as_tuple=False).squeeze(1)

    def encode_omics(self, omics_data: dict, node_type_tensor: torch.Tensor) -> torch.Tensor:
        """
        omics_data: {'rna': [Nr, Din_rna], 'methylation': [Nm, Din_met], 'snp': [Ns, Din_snp]}
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
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_omics(self, z: torch.Tensor, node_type_tensor: torch.Tensor) -> dict:

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
   
        h0 = self.encode_omics(omics_data, node_type_tensor)  # [N, n_hid]
        h = h0  
        if edge_index is not None and edge_index.numel() > 0: 
            for layer_idx, hgt_layer in enumerate(self.hgt_layers_list, start=1):
                h = hgt_layer(
                    node_inp=h,
                    node_type=node_type_tensor,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    edge_weights=edge_weights,
                    edge_sign=edge_sign,
                    edge_distance=edge_distance,
                )
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = self.reparameterize(mu, logvar)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

        recon = self.decode_omics(z, node_type_tensor)
        latent_dict = {"z": z}
        stats = {"z": z, "kl": kl}
        return latent_dict, recon, stats


class PoplarCrossOmicsModel(nn.Module):
    def __init__(self,
                 source_params: Dict[str, Dict],
                 node_type_mapping: Dict[str, int],
                 edge_type_mapping: Dict[str, int],
                 n_hid: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.2,
                 use_norm: bool = True,
                 adv_weight: float = 0.2,
                 mmd_weight: float = 0.1,
                 adv_lambda: float = 1.0,
                 hgt_layers: int = 2  
                 ):
        super().__init__()
        self.node_type_mapping = node_type_mapping
        self.edge_type_mapping = edge_type_mapping
        self.num_node_types = len(node_type_mapping)
        self.num_edge_relations = len(edge_type_mapping)

        self.graph_vae = GraphVAEWithHGT(
            source_params=source_params,
            node_type_mapping=self.node_type_mapping,
            num_edge_relations=self.num_edge_relations,
            n_hid=n_hid, 
            n_heads=n_heads,
            dropout=dropout, 
            use_norm=use_norm,
            hgt_layers=hgt_layers  
        )

        self.mmd = MMDLoss([1,2,4,8,16])   
        self.domain_adv = DomainAdversary(in_dim=n_hid, num_domains=self.num_node_types, hid=128, dropout=dropout)
        self.adv_weight = adv_weight
        self.mmd_weight = mmd_weight
        self.adv_lambda = adv_lambda
        self.kl_beta_start, self.kl_beta_end, self.kl_warmup_epochs = 0.0, 1.0, 10
        self.apply(init_weights)


    def compute_kl_beta(self, epoch: int) -> float:
        if epoch < self.kl_warmup_epochs:
            r = epoch / max(1, self.kl_warmup_epochs)
            return self.kl_beta_start + (self.kl_beta_end - self.kl_beta_start) * r
        return self.kl_beta_end

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

        kl_beta = self.compute_kl_beta(epoch)
        latent_dict, recon_dict, vae_stats = self.graph_vae(
            omics_data=omics_data_dict,
            node_type_tensor=node_type_tensor,
            edge_index=edge_index, edge_type=edge_type,
            edge_weights=edge_w, edge_sign=edge_s,
            edge_distance=None,
            kl_beta=kl_beta, logger=logger, epoch=epoch
        )

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


        adv_loss = torch.tensor(0.0, device=device)
        current_emb = vae_stats["z"]  
        if self.adv_weight > 0.0 and self.domain_adv is not None:
            dom_logits = self.domain_adv(current_emb, lambd=self.adv_lambda)
            targets = node_type_tensor.to(dom_logits.device, non_blocking=True)
            adv_loss = F.cross_entropy(dom_logits, targets)
            loss = loss + self.adv_weight * adv_loss

        z_snp = current_emb[(node_type_tensor == self.node_type_mapping["SNP"])]
        z_met = current_emb[(node_type_tensor == self.node_type_mapping["METH"])]
        z_rna = current_emb[(node_type_tensor == self.node_type_mapping["RNA"])]
        mmd_sr, _ = self.mmd(z_snp, z_rna)
        mmd_mr, _ = self.mmd(z_met, z_rna)

        mmd_loss = mmd_sr + mmd_mr
        loss = loss + self.mmd_weight* mmd_loss

        loss_dict = {
            "total_loss": loss,
            "reconstruction_loss": recon_loss,
            "recon_rna": recon_per_omics["rna"],
            "recon_methylation": recon_per_omics["methylation"],
            "recon_snp": recon_per_omics["snp"],
            "kl_loss": kl_loss,
            "mmd_loss": mmd_loss,
            "adv_loss": adv_loss if self.adv_weight > 0 else torch.tensor(0.0, device=device)
        }
        outputs = {
            "loss_dict": loss_dict,
            "node_emb": current_emb.detach(),  
            "node_out": current_emb.detach(), 
        }
        if return_post_hoc:
            outputs["post_hoc_scores"] = self.compute_post_hoc_edge_scores(current_emb, edge_indices_dict)
        return outputs

    @torch.no_grad()
    def compute_post_hoc_edge_scores(self, node_emb: torch.Tensor, edge_indices_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = F.normalize(node_emb, p=2, dim=-1)
        post = {}
        for rel_name, eidx in edge_indices_dict.items():
            if eidx is None or eidx.numel() == 0:
                post[rel_name] = torch.empty(0, device=node_emb.device); continue
            post[rel_name] = (z[eidx[0]] * z[eidx[1]]).sum(-1)
        return post
