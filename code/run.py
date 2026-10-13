import os
import random
import hashlib
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64;garbage_collection_threshold:0.6"
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from sklearn.decomposition import PCA
import json

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.preprocessing import StandardScaler

import gc
from model2 import *


def clean_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def stable_seed(seed, tag):
    digest = hashlib.sha256(f"{int(seed)}:{tag}".encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % (2**32)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Configuration

class ModelConfig:
    def __init__(self):

        self.source_params = {
            'rna': {'input_dim': 2000, 'hidden_dims': [128, 64], 'latent_dim': 32},
            'methylation': {'input_dim': 1500, 'hidden_dims': [128, 64], 'latent_dim': 32},
            'snp': {'input_dim': 1000, 'hidden_dims': [128, 64], 'latent_dim': 32}
        }

        self.node_type_mapping = {'SNP': 0, 'METH': 1, 'RNA': 2}


        self.n_hid = 64
        self.n_heads = 4
        self.hgt_layers = 2
        self.dropout = 0.2
        self.use_norm = True

        self.adv_weight = 0.1
        self.mmd_weight = 0.1
        self.adv_lambda = 1.0
        self.random_seed = 42


        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 100
        self.early_stopping_patience = 30


        self.kl_beta_start = 0.0
        self.kl_beta_end = 0.1
        self.kl_warmup_epochs = 300


        self.log_detailed_metrics_every = 1


class DataConfig:
    def __init__(self):
        self.node_type_mapping = {'SNP': 0, 'METH': 1, 'RNA': 2, 'unknown': 3}

# Data loading

class MultiOmicsDataLoader:
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.config = DataConfig()

    def set_seed(self, seed=42):
        set_global_seed(seed)

    def parse_graph_data(self, data_dict):
        print("Parsing saved graph data dictionary...")
        if 'edge_info_list' not in data_dict or len(data_dict['edge_info_list']) == 0:
            raise ValueError("edge_info_list not found or empty in saved data")
        edge_info_list = data_dict['edge_info_list']
        print(f"Found {len(edge_info_list)} edges in edge_info_list")
        for i, edge_info in enumerate(edge_info_list[:3]):
            print(f"Edge {i} sample: {edge_info}")

        edges_info = []
        for edge_info in edge_info_list:
            edges_info.append({
                'src_type': edge_info.get('src_type', 'unknown'),
                'src_id': edge_info.get('src_id', 'unknown'),
                'dst_type': edge_info.get('dst_type', 'unknown'),
                'dst_id': edge_info.get('dst_id', 'unknown'),
                'relation': edge_info.get('relation_type', 'unknown'),
                'weight': edge_info.get('weight', 1.0),
                'sign': edge_info.get('sign', 1),
            })




        edge_nodes = (
            {(str(e['src_type']), str(e['src_id'])) for e in edges_info}
            | {(str(e['dst_type']), str(e['dst_id'])) for e in edges_info}
        )
        universe_nodes = set()
        raw_universe = data_dict.get('node_universe')
        if raw_universe is not None:
            for item in raw_universe:
                if isinstance(item, dict):
                    nt = item.get('node_type', item.get('type'))
                    nid = item.get('node_id', item.get('id'))
                    if nt is not None and nid is not None:
                        universe_nodes.add((str(nt), str(nid)))
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    universe_nodes.add((str(item[0]), str(item[1])))
        all_nodes = list(edge_nodes | universe_nodes)

        def _count(nodes):
            d = {}
            for t, _ in nodes:
                d[t] = d.get(t, 0) + 1
            return d

        print(f"Found {len(all_nodes)} unique nodes")
        print(f"Node type distribution: {_count(all_nodes)}")
        w = [e['weight'] for e in edges_info]
        if w:
            print(f"Edge weights range: [{min(w):.4f}, {max(w):.4f}]")
        found_relations = sorted(list({e['relation'] for e in edges_info}))
        print(f"Found {len(found_relations)} relation types: {found_relations}")

        return {
            'nodes': all_nodes, 'edges_info': edges_info,
            'edge_weights': [e['weight'] for e in edges_info],
            'relations': found_relations,
            'node_type_counts': _count(all_nodes)
        }

    def load_omics_features(self):
        print("Loading original omics features...")
        features, shapes = {}, {}
        for feature_type in ['rna', 'methylation', 'snp']:
            df = pd.read_csv(self.data_paths[feature_type], index_col=0)
            df.index = df.index.astype(str)
            features[feature_type] = df
            shapes[feature_type] = df.shape
            print(f"{feature_type.upper()} features: {df.shape}, index examples: {df.index[:3].tolist()}")
        return features

    def create_node_mappings(self, graph_data, omics_features, add_orphan_nodes=True):
        """
        Build a node universe that is consistent with the graph AND omics matrices.

        Key design:
        - Graph IDs are kept as the canonical IDs for all graph nodes, so graph edges
          and held-out edge IDs resolve exactly to the node index.
        - CSV-only nodes are added only when their cleaned ID is not already represented
          by a graph node, preventing duplicate biological nodes caused by formatting
          differences such as case / '-' / '_'.
        - Graph-only nodes are retained (add_orphan_nodes=True) and their features are
          resolved by _clean_id() in create_omics_data_dict(); if truly unavailable,
          the existing zero-fill policy is used.
        """
        print("Creating node mappings (graph-preserving union of GRAPH and CSV).")
        nodes_by_type = {}

        graph_type_to_ids = {'RNA': [], 'METH': [], 'SNP': []}
        for t, i in graph_data['nodes']:
            t, i = str(t), str(i)
            if t in graph_type_to_ids:
                graph_type_to_ids[t].append(i)

        csv_ids = {
            'RNA':  [str(idx) for idx in omics_features['rna'].index.tolist()],
            'METH': [str(idx) for idx in omics_features['methylation'].index.tolist()],
            'SNP':  [str(idx) for idx in omics_features['snp'].index.tolist()],
        }

        def uniq_keep_order(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        for nt in ['SNP', 'METH', 'RNA']:
            graph_ids = uniq_keep_order(graph_type_to_ids[nt])
            csv_nt_ids = uniq_keep_order(csv_ids[nt])

            # Detect ambiguous cleaned IDs in the graph; silently merging them would
            # corrupt edge identities, so fail loudly instead.
            clean_to_graph = {}
            for gid in graph_ids:
                c = self._clean_id(gid)
                prev = clean_to_graph.get(c)
                if prev is not None and prev != gid:
                    raise RuntimeError(
                        f"Canonical node-ID collision in {nt}: {prev!r} and {gid!r} -> {c!r}"
                    )
                clean_to_graph[c] = gid

            if add_orphan_nodes:
                # Graph identity first. Add only genuinely new CSV biological nodes.
                order = list(graph_ids)
                graph_clean = set(clean_to_graph)
                for cid in csv_nt_ids:
                    if self._clean_id(cid) not in graph_clean:
                        order.append(cid)
            else:
                order = list(csv_nt_ids)

            order = uniq_keep_order(order)
            nodes_by_type[nt] = [(nt, nid) for nid in order]
            n_graph_only = sum(1 for gid in graph_ids if gid not in set(csv_nt_ids))
            print(
                f"{nt} nodes: total={len(order)}, graph={len(graph_ids)}, "
                f"csv={len(csv_nt_ids)}, graph-only(exact-ID)={n_graph_only}"
            )

        type_idx_mappings = {
            nt: {node: idx for idx, node in enumerate(lst)}
            for nt, lst in nodes_by_type.items()
        }
        return {'type_idx_mappings': type_idx_mappings, 'nodes_by_type': nodes_by_type}

    @staticmethod
    def _clean_id(id_str):
        if isinstance(id_str, str):
            return id_str.strip().lower().replace('-', '').replace('_', '').replace('(', '').replace(')', '')
        return str(id_str)

    def create_omics_data_dict(self, omics_features, mappings, fill_missing="zeros"):
        print("Creating omics data dictionary (align with nodes_by_type).")
        import numpy as np

        omics_to_node = {'rna': 'RNA', 'methylation': 'METH', 'snp': 'SNP'}
        omics_data_dict = {}

        for omics_type, node_type in omics_to_node.items():
            nodes = mappings['nodes_by_type'].get(node_type, [])
            if not nodes:
                omics_data_dict[omics_type] = torch.empty(0, 1, device=self.get_device())
                print(f"  {omics_type.upper()} has no node data; using an empty tensor")
                continue

            node_ids = [nid for _, nid in nodes]
            omics_df = omics_features[omics_type]

            col_mean = omics_df.mean(axis=0).values.astype(np.float32)
            pos = {self._clean_id(str(idx)): i for i, idx in enumerate(omics_df.index.tolist())}

            feats = []
            miss = 0
            for nid in node_ids:
                j = pos.get(self._clean_id(str(nid)))
                if j is not None:
                    feats.append(omics_df.iloc[j].values.astype(np.float32))
                else:
                    miss += 1
                    if fill_missing == "zeros":
                        feats.append(np.zeros(omics_df.shape[1], dtype=np.float32))
                    else:  # 'mean'
                        feats.append(col_mean)

            arr = np.stack(feats, axis=0)
            omics_data_dict[omics_type] = torch.from_numpy(arr)
            print(f"  {omics_type.upper()} mapped to {len(node_ids)} nodes, matched {len(node_ids)-miss}, filled {miss}; shape: {arr.shape}")

        return omics_data_dict

    def process_edges(self, graph_data, mappings):
        print("Processing edge information.")
        from collections import defaultdict
        edge_signs_dict = defaultdict(list)

        all_nodes_ordered = []
        node_to_global_idx = {}
        cur = 0
        for nt in ['SNP', 'METH', 'RNA']:
            for node in mappings['nodes_by_type'].get(nt, []):
                all_nodes_ordered.append(node)
                node_to_global_idx[node] = cur
                cur += 1
        global_idx_to_node_id = {idx: node_id for idx, (nt, node_id) in enumerate(all_nodes_ordered)}

        found_relations = sorted({e['relation'] for e in graph_data['edges_info']})
        edge_type_mapping = {rel: idx for idx, rel in enumerate(found_relations)}
        print(f"Edge type mapping: {edge_type_mapping}")


        edge_indices_dict = defaultdict(list)
        edge_weights_dict = defaultdict(list)

        for e in graph_data['edges_info']:
            src_node = (e['src_type'], e['src_id'])
            dst_node = (e['dst_type'], e['dst_id'])
            rel_type = e['relation']
            raw_sign = e.get('sign', 1)
            weight = float(e.get('weight', 1.0))

            if src_node not in node_to_global_idx or dst_node not in node_to_global_idx:
                continue
            if rel_type not in edge_type_mapping:
                continue



            if raw_sign == 0 or raw_sign < -1:
                sign = -2
            elif raw_sign > 1:
                sign = 1
            else:
                sign = int(raw_sign)


            src_global_idx = node_to_global_idx[src_node]
            dst_global_idx = node_to_global_idx[dst_node]
            edge_indices_dict[rel_type].append([src_global_idx, dst_global_idx])
            edge_weights_dict[rel_type].append(weight)
            edge_signs_dict[rel_type].append(sign)

        for rel_type in edge_indices_dict:
            edge_indices_dict[rel_type] = torch.LongTensor(edge_indices_dict[rel_type]).T
            edge_weights_dict[rel_type] = torch.FloatTensor(edge_weights_dict[rel_type])
            edge_signs_dict[rel_type] = torch.LongTensor(edge_signs_dict[rel_type])

        for rel_type in found_relations:
            if rel_type not in edge_indices_dict:
                edge_indices_dict[rel_type] = torch.empty((2, 0), dtype=torch.long)
                edge_weights_dict[rel_type] = torch.empty(0, dtype=torch.float)
                edge_signs_dict[rel_type] = torch.empty(0, dtype=torch.long)

        print(f"Processed edges summary:")
        for rel_type in edge_type_mapping:
            e_count = edge_indices_dict[rel_type].size(1)
            if e_count == 0:
                print(f"  • {rel_type}: 0 edges")
                continue
            weights = edge_weights_dict[rel_type]
            print(f"  • {rel_type}: {e_count} edges, weight range: [{weights.min():.4f}, {weights.max():.4f}]")

        return {
            'edge_indices_dict': edge_indices_dict,
            'edge_weights_dict': edge_weights_dict,
            'edge_signs_dict': edge_signs_dict,
            'edge_type_mapping': edge_type_mapping,
            'all_nodes_ordered': all_nodes_ordered,
            'node_to_global_idx': node_to_global_idx,
            'global_idx_to_node_id': global_idx_to_node_id,
            'valid_relations': list(found_relations)
        }

    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        print("Loading multi-omics data...")
        final = {}

        print("Step 1/4: Loading graph structure...")
        with open(self.data_paths['graph'], 'rb') as f:
            saved = pickle.load(f)
        raw_edge_info = saved['edge_info_list'] if isinstance(saved, dict) and 'edge_info_list' in saved else None
        if raw_edge_info is None:
            raise ValueError("Invalid graph data structure: missing 'edge_info_list'")
        graph_data = self.parse_graph_data({
            'edge_info_list': raw_edge_info,
            'node_universe': saved.get('node_universe') if isinstance(saved, dict) else None,
        })
        final['graph_data'] = graph_data

        print("\nStep 2/4: Loading omics features...")
        omics_features = self.load_omics_features()
        final['omics_features'] = omics_features

        print("\nStep 3/4: Creating node mappings...")
        mappings = self.create_node_mappings(graph_data, omics_features, add_orphan_nodes=True)

        final.update(mappings)

        print("\nStep 4/4: Creating omics data tensor...")
        omics_data_dict = self.create_omics_data_dict(omics_features, mappings)
        final['omics_data_dict'] = omics_data_dict

        print("\nStep 5/5: Processing edges and generating node_type_tensor...")
        edge_data = self.process_edges(graph_data, mappings)
        final.update(edge_data)

        node_type_list = []
        for node in final['all_nodes_ordered']:
            node_type = node[0]
            node_type_list.append(self.config.node_type_mapping.get(node_type, 3))  # 3=unknown
        final['node_type_tensor'] = torch.tensor(node_type_list, dtype=torch.long)
        print(f"Generated node_type_tensor: shape={final['node_type_tensor'].shape}, "
              f"type counts={torch.bincount(final['node_type_tensor'])}")

        print("\nData loading completed successfully!")
        return final

# Training

class GraphTrainer:
    def __init__(self, config, device, output_dir):
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []

        self.detailed_metrics = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': [],
            'kl_beta': [], 'recon_rna': [], 'recon_methylation': [], 'recon_snp': [],
            'mmd_loss': [], 'adv_loss': []
        }

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.setup_logging()

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"training_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training log saved to: {log_file}")
        self.logger.info("Model Configuration:")
        for k, v in vars(self.config).items():
            try:
                if isinstance(v, (dict, list)):
                    self.logger.info(f"  {k}: {json.dumps(v, indent=2, ensure_ascii=False, default=str)}")
                else:
                    self.logger.info(f"  {k}: {v}")
            except Exception:
                self.logger.info(f"  {k}: {v}")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_model(self, data):
        self.logger.info("Building PoplarCrossOmicsModel.")
        node_type_mapping = self.config.node_type_mapping
        edge_type_mapping = data['edge_type_mapping']

        self.model = PoplarCrossOmicsModel(
            source_params=self.config.source_params,
            node_type_mapping=node_type_mapping,
            edge_type_mapping=edge_type_mapping,
            n_hid=self.config.n_hid,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            use_norm=self.config.use_norm,
            adv_weight=self.config.adv_weight,
            mmd_weight=self.config.mmd_weight,
            adv_lambda=self.config.adv_lambda,
            hgt_layers=self.config.hgt_layers,
            random_seed=self.config.random_seed
        ).to(self.device)

        self.model.kl_beta_start = self.config.kl_beta_start
        self.model.kl_beta_end = self.config.kl_beta_end
        self.model.kl_warmup_epochs = self.config.kl_warmup_epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total model parameters: {total_params:,}")

    def _split_by_type(self, tensor_all, node_type_tensor):
        out = {}
        for name, tid in self.config.node_type_mapping.items():
            mask = (node_type_tensor == tid)
            out[name] = tensor_all[mask] if tensor_all.numel() > 0 and mask.any() else torch.empty(0, tensor_all.shape[1] if tensor_all.ndim == 2 else 0, device=tensor_all.device)
        return out

    def train_epoch(self, data, epoch):
        clean_cuda_cache()
        self.model.train()
        device = self.device


        omics_data = {k: v.to(device, non_blocking=True) for k, v in data['omics_data_dict'].items()}
        edge_indices_dict = {k: v.to(device, non_blocking=True) for k, v in data['edge_indices_dict'].items()}
        edge_weights_dict = {k: v.to(device, non_blocking=True) for k, v in data['edge_weights_dict'].items()}
        node_type_tensor = data['node_type_tensor'].to(device, non_blocking=True)
        edge_signs_dict = {k: v.to(device, non_blocking=True).long() for k, v in data.get('edge_signs_dict', {}).items()}


        for rel, eidx in edge_indices_dict.items():
            E_r = eidx.size(1)
            s = edge_signs_dict.get(rel, torch.empty(0, device=device, dtype=torch.long))
            if s.numel() != E_r:
                edge_signs_dict[rel] = torch.full((E_r,), -2, dtype=torch.long, device=device)

        self.logger.info("=" * 100)
        self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - training started (learning rate: {self.optimizer.param_groups[0]['lr']:.6f})")
        self.logger.info("=" * 100)


        self.logger.info(f"Epoch {epoch+1}: step 1/4 - forward pass")
        self.optimizer.zero_grad(set_to_none=True)
        model_out = self.model(
            omics_data_dict=omics_data,
            edge_indices_dict=edge_indices_dict,
            edge_weights_dict=edge_weights_dict,
            edge_signs_dict=edge_signs_dict,
            node_type_tensor=node_type_tensor,
            all_nodes_ordered=data['all_nodes_ordered'],
            epoch=epoch,
            return_post_hoc=False
        )


        loss_dict = model_out['loss_dict']
        total_loss = loss_dict['total_loss']
        recon_loss = loss_dict['reconstruction_loss']
        recon_rna = loss_dict.get('recon_rna', torch.tensor(0.0, device=device))
        recon_methylation = loss_dict.get('recon_methylation', torch.tensor(0.0, device=device))
        recon_snp = loss_dict.get('recon_snp', torch.tensor(0.0, device=device))
        kl_loss = loss_dict['kl_loss']
        mmd_loss = loss_dict.get('mmd_loss', torch.tensor(0.0, device=device))
        adv_loss = loss_dict.get('adv_loss', torch.tensor(0.0, device=device))
        kl_beta = float(self.model.compute_kl_beta(epoch))


        self.logger.info(f"Epoch {epoch+1}: step 2/4 - backward pass (KL beta: {kl_beta:.4f})")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()


        if (epoch + 1) % self.config.log_detailed_metrics_every == 0:
            self.detailed_metrics['total_loss'].append(total_loss.item())
            self.detailed_metrics['recon_loss'].append(recon_loss.item())
            self.detailed_metrics['kl_loss'].append(kl_loss.item())
            self.detailed_metrics['mmd_loss'].append(mmd_loss.item())
            self.detailed_metrics['adv_loss'].append(adv_loss.item())
            self.detailed_metrics['kl_beta'].append(kl_beta)
            self.detailed_metrics['recon_rna'].append(recon_rna.item())
            self.detailed_metrics['recon_methylation'].append(recon_methylation.item())
            self.detailed_metrics['recon_snp'].append(recon_snp.item())

            self.logger.info("=" * 80)
            self.logger.info(f"Epoch {epoch+1} Loss Metrics")
            self.logger.info("=" * 80)
            self.logger.info(f"  Total Loss: {total_loss.item():.6f}")
            self.logger.info(f"  Total Reconstruction Loss: {recon_loss.item():.6f}")
            self.logger.info(f"    RNA Reconstruction Loss: {recon_rna.item():.6f}")
            self.logger.info(f"    Methylation Reconstruction Loss: {recon_methylation.item():.6f}")
            self.logger.info(f"    SNP Reconstruction Loss: {recon_snp.item():.6f}")
            self.logger.info(f"  KL Loss: {kl_loss.item():.6f}")
            self.logger.info(f"  MMD Loss: {mmd_loss.item():.6f}")
            self.logger.info(f"  Adversarial Loss: {adv_loss.item():.6f}")
            self.logger.info(f"  KL Beta: {kl_beta:.4f}")
            self.logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.logger.info("=" * 80)

        torch.cuda.empty_cache()
        self.logger.info(f"Epoch {epoch+1}: step 4/4 - complete\n")
        clean_cuda_cache()
        return total_loss.item(), loss_dict, {}

    def train(self, data):
        self.logger.info("Starting training.")
        pbar = tqdm(range(self.config.num_epochs), desc="Training Progress")
        best_epoch_output = None
        for epoch in pbar:
            loss, loss_dict, current = self.train_epoch(data, epoch)
            self.train_losses.append(loss)
            self.scheduler.step(loss)
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
                self.save_model(os.path.join(self.output_dir, 'best_model.pth'))
                best_epoch_output = current
                self.logger.info(f"Epoch {epoch+1}: New best loss {loss:.6f} → Saved best model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered! Best loss: {self.best_loss:.6f}")
                    break
            pbar.set_description(f"Epoch {epoch+1}/{self.config.num_epochs}: Total Loss = {loss:.4f}")

        self.load_model(os.path.join(self.output_dir, 'best_model.pth'))
        self.save_epoch_loss_detail()
        self.plot_training_curve()
        self.plot_training_curves_advanced()

        self.logger.info("Training completed!")
        return best_epoch_output or {}

    def save_model(self, path):
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'detailed_metrics': self.detailed_metrics
        }
        torch.save(ckpt, path)
        self.logger.info(f"Model saved to: {path}")

    def load_model(self, path):
        if not os.path.exists(path):
            self.logger.warning(f"Model file not found: {path}")
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_losses = ckpt.get('train_losses', [])
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.detailed_metrics = ckpt.get('detailed_metrics', self.detailed_metrics)
        self.logger.info(f"Model loaded from: {path}")
        return True

    def plot_training_curve(self, save_path='training_pngcurve.png'):
        if not self.train_losses:
            self.logger.warning("No training loss data to plot")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses)+1), self.train_losses, alpha=0.7, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        full_path = os.path.join(self.output_dir, save_path)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Training curve saved to: {full_path}")

    def plot_training_curves_advanced(self):
        import numpy as np
        stats_dir = os.path.join(self.output_dir, "training_stats")
        os.makedirs(stats_dir, exist_ok=True)

        dm = self.detailed_metrics
        epochs = np.arange(1, len(dm['total_loss']) + 1)
        if len(epochs) == 0:
            self.logger.warning("No loss data for advanced plot")
            return

        series = {
            'Total Loss'   : dm['total_loss'],
            'Recon'        : dm['recon_loss'],
            'KL'           : dm['kl_loss'],
            'MMD'          : dm['mmd_loss'],
            'Adversarial'  : dm['adv_loss'],
        }


        plt.figure(figsize=(12, 6))
        for k, v in series.items():
            plt.plot(epochs, v, label=k, alpha=0.85, linewidth=1.8)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log-scale)')
        plt.title('Training losses (log scale)')
        plt.grid(alpha=0.3)
        plt.legend(ncol=3)
        plt.tight_layout()
        p1 = os.path.join(stats_dir, "loss_logscale.png")
        plt.savefig(p1, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved {p1}")


        vals_max = np.array([np.nanmax(v) if len(v)>0 else np.nan for v in series.values()])
        if np.nanmax(vals_max) / max(np.nanmin(vals_max), 1e-8) >= 50:
            med = np.nanmedian([np.nanmedian(v) for v in series.values()])
            left_keys, right_keys = [], []
            for k, v in series.items():
                (left_keys if np.nanmedian(v) >= med else right_keys).append(k)

            fig, ax = plt.subplots(figsize=(12, 6))
            for k in left_keys:
                ax.plot(epochs, series[k], label=k, linewidth=2.2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (left)')
            ax.grid(alpha=0.3)

            ax2 = ax.twinx()
            for k in right_keys:
                ax2.plot(epochs, series[k], '--', label=k, linewidth=1.8)
            ax2.set_ylabel('Loss (right)')

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper right', ncol=2)
            plt.title('Training losses (dual y-axes)')
            plt.tight_layout()
            p2 = os.path.join(stats_dir, "loss_dual_axes.png")
            plt.savefig(p2, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved {p2}")

    def save_epoch_loss_detail(self):
        stats_dir = os.path.join(self.output_dir, "training_stats")
        os.makedirs(stats_dir, exist_ok=True)
        csv_path = os.path.join(stats_dir, "epoch_loss_detail.csv")

        total_epochs = len(self.detailed_metrics['total_loss'])
        if total_epochs == 0:
            self.logger.warning("No loss data to save (detailed_metrics is empty)")
            return

        rows = []
        for epoch in range(total_epochs):
            rows.append({
                'epoch': epoch + 1,
                'total_loss': self.detailed_metrics['total_loss'][epoch],
                'recon_loss': self.detailed_metrics['recon_loss'][epoch],
                'kl_loss': self.detailed_metrics['kl_loss'][epoch],
                'kl_beta': self.detailed_metrics['kl_beta'][epoch],
                'mmd_loss': self.detailed_metrics['mmd_loss'][epoch],
                'adv_loss': self.detailed_metrics['adv_loss'][epoch],
                'recon_rna': self.detailed_metrics['recon_rna'][epoch],
                'recon_methylation': self.detailed_metrics['recon_methylation'][epoch],
                'recon_snp': self.detailed_metrics['recon_snp'][epoch]
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
        self.logger.info(f"Saved epoch loss detail to: {csv_path}")

# Data preprocessing

def preprocess_omics_data(data):
    print("\nPreprocessing multi-omics data (unit variance)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def standardize(tensor, dim, clip_extreme=False, clip_percentile=(1, 99)):
        if tensor.numel() == 0:
            return tensor
        x = tensor
        if clip_extreme:

            lower = torch.quantile(x, clip_percentile[0] / 100.0, dim=dim, keepdim=True)
            upper = torch.quantile(x, clip_percentile[1] / 100.0, dim=dim, keepdim=True)
            x = torch.clamp(x, lower, upper)
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        x = (x - mean) / std

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    rna_raw = data['omics_data_dict']['rna']
    if rna_raw.numel() > 0:
        rna_std = standardize(torch.log1p(rna_raw), dim=1, clip_extreme=False)
        data['omics_data_dict']['rna'] = rna_std
        glob_var = float(rna_std.var())
        print(f"RNA: standardized range={rna_std.min():.2f}~{rna_std.max():.2f}, global variance~{glob_var:.2f}")


    meth_raw = data['omics_data_dict']['methylation']
    if meth_raw.numel() > 0:
        x = meth_raw.clone()

        if float(x.max()) > 1.05:
            x = x / 100.0


        eps = 1e-6
        x = torch.clamp(x, eps, 1 - eps)
        x = torch.logit(x, eps)


        x = standardize(x, dim=0, clip_extreme=True, clip_percentile=(1, 99))
        data['omics_data_dict']['methylation'] = x
        glob_var = float(x.var())
        print(f"METH: standardized range={x.min():.2f}~{x.max():.2f}, global variance~{glob_var:.2f}")

    snp_raw = data['omics_data_dict']['snp']
    if snp_raw.numel() > 0:
        snp_std = standardize(torch.log1p(snp_raw), dim=1, clip_extreme=False)
        data['omics_data_dict']['snp'] = snp_std
        glob_var = float(snp_std.var())
        print(f"SNP: standardized range={snp_std.min():.2f}~{snp_std.max():.2f}, global variance~{glob_var:.2f}")

    return data

# Result export

@torch.no_grad()
def save_final_results(
    trainer, data, output_dir, eval_seed=42,
    knn_k=30, trust_max_samples=None,
):
    trainer.model.eval()

    set_global_seed(eval_seed)

    device = trainer.device
    results_dir = os.path.join(output_dir, "final_results")
    os.makedirs(results_dir, exist_ok=True)


    omics_data = {k: v.to(device, non_blocking=True) for k, v in data['omics_data_dict'].items()}

    node_type_tensor = data['node_type_tensor'].to(device, non_blocking=True)

    edge_signs_dict = {k: v.to(device, non_blocking=True) for k, v in data['edge_signs_dict'].items()}


    model_out = trainer.model(
        omics_data_dict=omics_data,
        edge_indices_dict=data['edge_indices_dict'],
        edge_weights_dict=data['edge_weights_dict'],
        edge_signs_dict=edge_signs_dict,
        node_type_tensor=node_type_tensor,
        all_nodes_ordered=data['all_nodes_ordered'],
        epoch=0,
        return_post_hoc=False
    )
    node_out = model_out['node_out']


    loss_dict = model_out.get('loss_dict', {})

    def _loss_value(name):
        x = loss_dict.get(name, None)
        if x is None:
            return float('nan')
        return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)

    reconstruction_summary = {
        'reconstruction_total_mse': _loss_value('reconstruction_loss'),
        'reconstruction_rna_mse': _loss_value('recon_rna'),
        'reconstruction_methylation_mse': _loss_value('recon_methylation'),
        'reconstruction_snp_mse': _loss_value('recon_snp'),
    }
    pd.DataFrame([reconstruction_summary]).to_csv(
        os.path.join(results_dir, 'reconstruction_metrics.csv'), index=False
    )

    def save_embedding(emb_tensor, save_dir_name, prefix):
        emb_dir = os.path.join(results_dir, save_dir_name)
        os.makedirs(emb_dir, exist_ok=True)
        for node_type_name, node_type_id in trainer.config.node_type_mapping.items():
            mask = (node_type_tensor == node_type_id)
            if not mask.any() or emb_tensor[mask].numel() == 0:
                trainer.logger.warning(f"No valid {node_type_name} nodes for {prefix} embedding, skip saving")
                continue

            emb = emb_tensor[mask].cpu().numpy()
            emb_df = pd.DataFrame(emb, columns=[f"{prefix}_dim_{i}" for i in range(emb.shape[1])])
            node_global_indices = torch.where(mask)[0].cpu().tolist()
            node_ids = [data['all_nodes_ordered'][idx][1] for idx in node_global_indices]

            emb_df.insert(0, "node_id", node_ids)
            emb_df.insert(1, "node_type", node_type_name)

            save_path = os.path.join(emb_dir, f"{node_type_name.lower()}_{prefix}_embeddings.csv")
            emb_df.to_csv(save_path, index=False, float_format="%.6f")
            trainer.logger.info(f"Saved {node_type_name} {prefix} embedding to: {save_path}")

    save_embedding(emb_tensor=node_out, save_dir_name="node_embeddings", prefix="node_out")

    sim_save_dir = os.path.join(results_dir, "cosine_similarities")
    os.makedirs(sim_save_dir, exist_ok=True)
    node_emb_normalized = F.normalize(node_out, p=2, dim=-1)
    global_idx_to_node_id = data['global_idx_to_node_id']

    for rel_name, eidx in data['edge_indices_dict'].items():
        if eidx.numel() == 0:
            trainer.logger.warning(f"Empty edges for relation {rel_name}, skip similarity calculation")
            continue

        src_global_idx = eidx[0].to(device)
        dst_global_idx = eidx[1].to(device)

        edge_weights = data['edge_weights_dict'].get(rel_name, torch.ones(eidx.size(1), device=device)).to(device)
        edge_signs_tensor = data['edge_signs_dict'].get(rel_name, torch.full((eidx.size(1),), -2, dtype=torch.long, device=device))

        src_emb = node_emb_normalized[src_global_idx]
        dst_emb = node_emb_normalized[dst_global_idx]
        cosine_sim = torch.sum(src_emb * dst_emb, dim=-1).cpu().numpy()

        src_node_ids = [global_idx_to_node_id.get(idx.item(), "unknown") for idx in src_global_idx.cpu()]
        dst_node_ids = [global_idx_to_node_id.get(idx.item(), "unknown") for idx in dst_global_idx.cpu()]
        sim_df = pd.DataFrame({
            "src_node_id": src_node_ids,
            "dst_node_id": dst_node_ids,
            "relation_name": rel_name,
            "relation_id": trainer.model.edge_type_mapping[rel_name],
            "edge_sign": edge_signs_tensor.cpu().numpy(),
            "original_edge_weight": edge_weights.cpu().numpy(),
            "node_out_cosine_similarity": cosine_sim
        })
        sim_save_path = os.path.join(sim_save_dir, f"{rel_name}_cosine_similarity.csv")
        sim_df.to_csv(sim_save_path, index=False, float_format="%.6f")
        trainer.logger.info(f"Saved {rel_name} cosine similarity (n={len(sim_df)}) to: {sim_save_path}")

    binc = torch.bincount(data['node_type_tensor'])
    node_type_counts = {str(i): int(binc[i].item()) for i in range(len(binc))}
    summary = {
        'Training Info': {
            'Total Epochs': len(trainer.train_losses),
            'Best Loss': trainer.best_loss,
            'Early Stopping Triggered': trainer.patience_counter >= trainer.config.early_stopping_patience,
            'Final KL Beta': trainer.model.compute_kl_beta(len(trainer.train_losses))
        },
        'Data Stats': {
            'Total Nodes': len(data['all_nodes_ordered']),
            'Node Type Counts': node_type_counts,
            'Valid Relations': data['valid_relations'],
            'Omics Data Shapes': {k: tuple(v.shape) for k, v in data['omics_data_dict'].items()},
            'Saved Embeddings': {

                'node_out': f"Shape: {node_out.shape}, Dim: {node_out.shape[1]} (VAE latent vector, no extra HGT/projection)"
            }
        },
        'Model Config': vars(trainer.config)
    }
    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    trainer.logger.info(f"Saved training summary to: {summary_path}")

    alignment_summary = compute_alignment_metrics(
        trainer, data, output_dir, node_out=node_out, which="node_out",
        knn_k=knn_k, trust_max_samples=trust_max_samples,
    )

    return node_out.detach().cpu(), alignment_summary, reconstruction_summary


def compute_alignment_metrics(
    trainer,
    data,
    output_dir,
    node_out,
    which="node_out",
    knn_k=30,
    trust_max_samples=None,
):
    results_dir = os.path.join(output_dir, "final_results", "alignment_metrics", which)
    os.makedirs(results_dir, exist_ok=True)

    Z_all_t = node_out.detach().cpu()
    node_type_tensor = data['node_type_tensor'].detach().cpu()

    type_to_cfg = {
        'RNA': ('rna', trainer.config.node_type_mapping['RNA']),
        'METH': ('methylation', trainer.config.node_type_mapping['METH']),
        'SNP': ('snp', trainer.config.node_type_mapping['SNP']),
    }

    typed = {}
    offset_ids = np.asarray([str(nid) for _, nid in data['all_nodes_ordered']], dtype=object)
    for tname, (omics_key, tid) in type_to_cfg.items():
        mask = (node_type_tensor == tid).numpy()
        typed[tname] = {
            'ids': offset_ids[mask],
            'Z': Z_all_t[torch.from_numpy(mask)].numpy(),
            'X': data['omics_data_dict'][omics_key].detach().cpu().numpy(),
        }
        if typed[tname]['Z'].shape[0] != typed[tname]['X'].shape[0]:
            raise RuntimeError(
                f"{tname} X/Z row mismatch: X={typed[tname]['X'].shape}, Z={typed[tname]['Z'].shape}"
            )

    # ---------- iLISI: exclude the query node itself ----------
    Z_all = np.vstack([typed['RNA']['Z'], typed['METH']['Z'], typed['SNP']['Z']])
    labels = np.asarray(
        ['RNA'] * len(typed['RNA']['Z']) +
        ['METH'] * len(typed['METH']['Z']) +
        ['SNP'] * len(typed['SNP']['Z'])
    )
    if len(labels) < 3:
        raise RuntimeError('Too few nodes for iLISI/domain metrics.')
    kk = min(int(knn_k), len(labels) - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine").fit(Z_all)
    _, idx = nn.kneighbors(Z_all, return_distance=True)
    idx = idx[:, 1:]  # remove self neighbor
    mods = np.unique(labels)
    ilisi = []
    for i in range(Z_all.shape[0]):
        neigh = labels[idx[i]]
        pm = np.asarray([(neigh == mod).mean() for mod in mods], dtype=float)
        ilisi.append(1.0 / np.sum(pm ** 2))
    ilisi = np.asarray(ilisi, dtype=float)
    pd.DataFrame({
        "node_index": np.arange(len(ilisi)),
        "iLISI": ilisi,
        "modality": labels,
    }).to_csv(os.path.join(results_dir, "ilisi_per_node.csv"), index=False)

    # ---------- Domain accuracy: 5-fold CV is internal estimation, not 5 experimental repeats ----------
    if len(np.unique(labels)) >= 2:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tr, te in skf.split(Z_all, labels):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Z_all[tr], labels[tr])
            accs.append(accuracy_score(labels[te], clf.predict(Z_all[te])))
        domain_acc = float(np.mean(accs))
        domain_cv_sd = float(np.std(accs, ddof=1))
    else:
        accs = []
        domain_acc = float('nan')
        domain_cv_sd = float('nan')
    pd.DataFrame({"fold": np.arange(1, len(accs) + 1), "accuracy": accs}).to_csv(
        os.path.join(results_dir, "domain_accuracy_cv5_folds.csv"), index=False
    )

    # ---------- Trustworthiness: per modality in its own original feature space, then weighted mean ----------
    # Different modalities have different X dimensions, so they cannot be naively concatenated.
    trust_rows = []
    weighted_num = 0.0
    weighted_den = 0
    for tname, (omics_key, _) in type_to_cfg.items():
        X = np.asarray(typed[tname]['X'], dtype=np.float64)
        Z = np.asarray(typed[tname]['Z'], dtype=np.float64)
        n = X.shape[0]
        if n < 4:
            tw = float('nan')
            n_used = n
            k_tw = 0
        else:
            indices = np.arange(n)
            if trust_max_samples is not None and trust_max_samples > 0 and n > trust_max_samples:
                rng = np.random.default_rng(20260905)
                indices = np.sort(rng.choice(n, size=int(trust_max_samples), replace=False))
                X = X[indices]
                Z = Z[indices]
            n_used = X.shape[0]
            ncomp = min(50, X.shape[1], n_used - 1)
            if ncomp >= 1:
                Xp = PCA(n_components=ncomp, svd_solver='auto', random_state=42).fit_transform(X)
            else:
                Xp = X
            Xp = StandardScaler().fit_transform(Xp)
            k_tw = min(int(knn_k), max(1, n_used // 2 - 1))
            tw = float(sklearn_trustworthiness(Xp, Z, n_neighbors=k_tw, metric='euclidean'))
            weighted_num += tw * n_used
            weighted_den += n_used
        trust_rows.append({
            'modality': tname,
            'n_nodes': int(n),
            'n_used': int(n_used),
            'k': int(k_tw),
            'trustworthiness': tw,
        })
    trust_df = pd.DataFrame(trust_rows)
    trust_df.to_csv(os.path.join(results_dir, 'trustworthiness_by_modality.csv'), index=False)
    trust_overall = float(weighted_num / weighted_den) if weighted_den > 0 else float('nan')

    summary = {
        "which_embedding": which,
        "ilisi_mean": float(np.nanmean(ilisi)) if len(ilisi) else float('nan'),
        "ilisi_sd_across_nodes": float(np.nanstd(ilisi, ddof=1)) if len(ilisi) > 1 else 0.0,
        "ilisi_max_possible": int(len(mods)),
        "domain_classifier_accuracy_cv5": domain_acc,
        "domain_classifier_cv_fold_sd": domain_cv_sd,
        "trustworthiness_weighted_mean": trust_overall,
        "trustworthiness_by_modality": {
            r['modality']: float(r['trustworthiness']) for r in trust_rows
        },
        "trustworthiness_n_used_by_modality": {
            r['modality']: int(r['n_used']) for r in trust_rows
        },
    }
    with open(os.path.join(results_dir, "metrics_summary.json"), "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # One-run scalar table for quick inspection.
    scalar_df = pd.DataFrame([{
        'domain_accuracy_cv5': domain_acc,
        'iLISI_mean': summary['ilisi_mean'],
        'trustworthiness_weighted_mean': trust_overall,
        'trustworthiness_RNA': summary['trustworthiness_by_modality'].get('RNA', np.nan),
        'trustworthiness_METH': summary['trustworthiness_by_modality'].get('METH', np.nan),
        'trustworthiness_SNP': summary['trustworthiness_by_modality'].get('SNP', np.nan),
    }])
    scalar_df.to_csv(os.path.join(results_dir, 'scalar_metrics.csv'), index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(ilisi, bins=30)
    plt.xlabel("iLISI")
    plt.ylabel("Count")
    plt.title("iLISI distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ilisi_hist.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.bar([0], [domain_acc])
    plt.ylim(0, 1)
    plt.xticks([0], ["LR CV-acc"])
    plt.ylabel("Accuracy")
    plt.title("Domain separability")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "domain_acc.png"), dpi=300)
    plt.close()

    trainer.logger.info(f"Alignment/representation metrics saved to {results_dir}")
    return summary

# Visualization

def create_visualizations(trainer, data, output_dir, node_out=None, eval_seed=42):
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    device = trainer.device

    if node_out is not None:
        model_out = {'node_out': node_out.to(device)}
    else:
        # Recompute embeddings when node_out is not provided.
        set_global_seed(eval_seed)
        trainer.model.eval()
        with torch.no_grad():
            omics_data = {k: v.to(device) for k, v in data['omics_data_dict'].items()}
            edge_indices_dict = {k: v.to(device) for k, v in data['edge_indices_dict'].items()}
            edge_weights_dict = {k: v.to(device) for k, v in data['edge_weights_dict'].items()}
            node_type_tensor = data['node_type_tensor'].to(device)
            edge_signs_dict = {k: v.to(device) for k, v in data['edge_signs_dict'].items()}

            model_out = trainer.model(
                omics_data_dict=omics_data,
                edge_indices_dict=edge_indices_dict,
                edge_weights_dict=edge_weights_dict,
                edge_signs_dict=edge_signs_dict,
                node_type_tensor=node_type_tensor,
                all_nodes_ordered=data['all_nodes_ordered'],
                return_post_hoc=False
            )


    plt.figure(figsize=(15, 5))
    node_types = ['SNP', 'METH', 'RNA']
    for idx, node_type in enumerate(node_types):
        tid = trainer.config.node_type_mapping[node_type]
        mask = (data['node_type_tensor'] == tid)
        plt.subplot(1, 3, idx+1)
        if mask.sum() < 2:
            plt.title(f"{node_type} Embeddings (Insufficient Samples)")
            continue

        emb = model_out['node_out'][mask]
        if emb.numel() == 0:
            plt.title(f"{node_type} Embeddings (Empty)")
            continue

        emb_np = emb.cpu().numpy()
        pca = PCA(n_components=2, random_state=42)
        emb_2d = pca.fit_transform(emb_np)

        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=20)
        plt.title(f"{node_type} Embeddings (PCA)\nVar Explained: {pca.explained_variance_ratio_.sum():.2%}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "node_embeddings_pca.png"), dpi=300, bbox_inches='tight')
    plt.close()
    trainer.logger.info(f"Saved node embeddings PCA plot to {vis_dir}")


    if len(trainer.detailed_metrics['total_loss']) > 0:
        epochs = range(1, len(trainer.detailed_metrics['total_loss']) + 1)
        recon_losses = trainer.detailed_metrics['recon_loss']
        kl_losses = trainer.detailed_metrics['kl_loss']
        mmd_losses = trainer.detailed_metrics['mmd_loss']
        adv_losses = trainer.detailed_metrics['adv_loss']

        plt.figure(figsize=(12, 6))
        plt.stackplot(
            epochs, recon_losses, kl_losses, mmd_losses, adv_losses,
            labels=['Reconstruction Loss', 'KL Loss', 'MMD Loss', 'Adversarial Loss'],
            alpha=0.7
        )
        plt.plot(epochs, trainer.detailed_metrics['total_loss'], linewidth=2, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Breakdown During Training')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "loss_breakdown.png"), dpi=300, bbox_inches='tight')
        plt.close()
        trainer.logger.info(f"Saved loss breakdown plot to {vis_dir}")

# Held-out edge evaluation
DEFAULT_GRAPH_ROOT = r"graph"
DEFAULT_HELDOUT_ROOT = os.path.join(DEFAULT_GRAPH_ROOT, "heldout")
DEFAULT_OUTPUT_ROOT = r"/result"


def canonical_edge(edge):
    return (
        str(edge.get('src_type', '')),
        str(edge.get('src_id', '')),
        str(edge.get('dst_type', '')),
        str(edge.get('dst_id', '')),
        str(edge.get('relation_type', edge.get('relation', ''))),
    )


def load_hidden_payload(hidden_path):
    with open(hidden_path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and 'edges_by_relation' in obj:
        return obj.get('meta', {}), obj['edges_by_relation']

    if isinstance(obj, dict):
        edges = {k: v for k, v in obj.items() if isinstance(v, list)}
        return {}, edges

    raise ValueError(f"Invalid hidden edge file: {hidden_path}")


def resolve_experiment_paths(args):
    """
    A single experiment/seed resolves graph, hidden edges and output directory together.
    This is the key mechanism that prevents mask-seed/train-seed mismatch.
    """
    if args.experiment == 'full':
        graph_path = os.path.join(args.graph_root, 'guide_graph_pruned.pkl')
        hidden_path = None
        manifest_path = None
    else:
        exp_dir = os.path.join(args.heldout_root, args.experiment, f'seed_{args.seed}')
        graph_path = os.path.join(exp_dir, 'guide_graph_masked.pkl')
        hidden_path = os.path.join(exp_dir, 'hidden_edges.pkl')
        manifest_path = os.path.join(exp_dir, 'manifest.json')

    output_dir = os.path.join(args.output_root, args.experiment, f'seed_{args.seed}')
    return graph_path, hidden_path, manifest_path, output_dir


def validate_experiment_binding(experiment, seed, graph_path, hidden_path, manifest_path):
    """Hard-fail if a masked graph/hidden-edge pair does not belong to this run."""
    if experiment == 'full':
        return None

    for path, label in [(graph_path, 'masked graph'), (hidden_path, 'hidden edges'), (manifest_path, 'manifest')]:
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    if str(manifest.get('experiment')) != str(experiment):
        raise RuntimeError(
            f"Experiment mismatch: CLI={experiment}, manifest={manifest.get('experiment')}"
        )
    if int(manifest.get('experiment_seed')) != int(seed):
        raise RuntimeError(
            f"Seed mismatch: train seed={seed}, mask seed={manifest.get('experiment_seed')}"
        )

    expected_graph = os.path.realpath(manifest.get('masked_graph', graph_path))
    expected_hidden = os.path.realpath(manifest.get('hidden_edges', hidden_path))
    if os.path.realpath(graph_path) != expected_graph:
        raise RuntimeError(f"Masked graph path does not match manifest: {graph_path}")
    if os.path.realpath(hidden_path) != expected_hidden:
        raise RuntimeError(f"Hidden-edge path does not match manifest: {hidden_path}")

    hidden_meta, _ = load_hidden_payload(hidden_path)
    if hidden_meta:
        if str(hidden_meta.get('experiment')) != str(experiment):
            raise RuntimeError("hidden_edges.pkl experiment metadata does not match CLI.")
        if int(hidden_meta.get('experiment_seed')) != int(seed):
            raise RuntimeError("hidden_edges.pkl seed metadata does not match train seed.")

    return manifest


def verify_no_hidden_edge_leakage(data, hidden_by_relation):
    """Check that held-out positives are truly absent from the graph used for training."""
    train_keys = {canonical_edge(e) for e in data['graph_data']['edges_info']}
    hidden_keys = set()
    for relation, edges in hidden_by_relation.items():
        for e in edges:
            hidden_keys.add(canonical_edge(e))
    overlap = train_keys & hidden_keys
    if overlap:
        example = next(iter(overlap))
        raise RuntimeError(f"Held-out edge leakage detected before evaluation: {example}")


@torch.no_grad()
def evaluate_hidden_edges(
    node_out,
    data,
    hidden_path,
    output_dir,
    seed,
    threshold=0.5,
    null_permutations=100,
):
    hidden_meta, hidden_by_relation = load_hidden_payload(hidden_path)
    verify_no_hidden_edge_leakage(data, hidden_by_relation)

    eval_dir = os.path.join(output_dir, 'final_results', 'heldout_edge_recovery')
    os.makedirs(eval_dir, exist_ok=True)

    node_to_idx = {
        (str(node_type), str(node_id)): int(idx)
        for idx, (node_type, node_id) in enumerate(data['all_nodes_ordered'])
    }

    # Canonical fallback catches pure formatting mismatches while refusing ambiguous merges.
    canonical_to_idx = {}
    canonical_collision = {}
    for (nt, nid), idx in node_to_idx.items():
        key = (nt, MultiOmicsDataLoader._clean_id(nid))
        if key in canonical_to_idx and canonical_to_idx[key] != idx:
            canonical_collision.setdefault(key, []).append((nid, idx))
        else:
            canonical_to_idx[key] = idx
    if canonical_collision:
        examples = list(canonical_collision.items())[:3]
        raise RuntimeError(f'Ambiguous canonical node IDs in embedding index: {examples}')

    graph_nodes = {
        (str(t), str(i)) for t, i in data['graph_data'].get('nodes', [])
    }
    graph_nodes_canon = {
        (t, MultiOmicsDataLoader._clean_id(i)) for t, i in graph_nodes
    }

    feature_nodes_canon = {}
    feature_map = {'RNA': 'rna', 'METH': 'methylation', 'SNP': 'snp'}
    for nt, omics_name in feature_map.items():
        df = data['omics_features'][omics_name]
        feature_nodes_canon[nt] = {
            MultiOmicsDataLoader._clean_id(str(i)) for i in df.index.tolist()
        }

    z = F.normalize(node_out.float(), p=2, dim=1).cpu()
    rng = np.random.default_rng(stable_seed(seed, 'heldout_pairing_null'))

    metric_rows = []
    missing_records = []

    def resolve_idx(nt, nid):
        exact = node_to_idx.get((nt, nid))
        if exact is not None:
            return exact, 'exact'
        canon = canonical_to_idx.get((nt, MultiOmicsDataLoader._clean_id(nid)))
        if canon is not None:
            return canon, 'canonical_fallback'
        return None, 'missing'

    for relation in sorted(hidden_by_relation.keys()):
        hidden_edges = hidden_by_relation[relation]
        n_hidden = len(hidden_edges)

        src_indices, dst_indices = [], []
        evaluable_edges = []
        relation_missing = 0

        for edge in hidden_edges:
            src_type, src_id = str(edge.get('src_type')), str(edge.get('src_id'))
            dst_type, dst_id = str(edge.get('dst_type')), str(edge.get('dst_id'))
            src_idx, src_match = resolve_idx(src_type, src_id)
            dst_idx, dst_match = resolve_idx(dst_type, dst_id)

            if src_idx is None or dst_idx is None:
                relation_missing += 1
                src_c = MultiOmicsDataLoader._clean_id(src_id)
                dst_c = MultiOmicsDataLoader._clean_id(dst_id)
                if src_idx is None and dst_idx is None:
                    reason = 'both_missing'
                elif src_idx is None:
                    reason = 'source_missing'
                else:
                    reason = 'target_missing'

                missing_records.append({
                    'relation': relation,
                    'src_type': src_type,
                    'src_id': src_id,
                    'src_in_masked_graph_exact': (src_type, src_id) in graph_nodes,
                    'src_in_masked_graph_canonical': (src_type, src_c) in graph_nodes_canon,
                    'src_in_feature_matrix_canonical': src_c in feature_nodes_canon.get(src_type, set()),
                    'src_exists_in_embedding': src_idx is not None,
                    'dst_type': dst_type,
                    'dst_id': dst_id,
                    'dst_in_masked_graph_exact': (dst_type, dst_id) in graph_nodes,
                    'dst_in_masked_graph_canonical': (dst_type, dst_c) in graph_nodes_canon,
                    'dst_in_feature_matrix_canonical': dst_c in feature_nodes_canon.get(dst_type, set()),
                    'dst_exists_in_embedding': dst_idx is not None,
                    'reason': reason,
                })
                continue

            src_indices.append(src_idx)
            dst_indices.append(dst_idx)
            evaluable_edges.append((edge, src_match, dst_match))

        n_evaluable = len(src_indices)
        if n_evaluable:
            src_t = torch.as_tensor(src_indices, dtype=torch.long)
            dst_t = torch.as_tensor(dst_indices, dtype=torch.long)
            raw = torch.sum(z[src_t] * z[dst_t], dim=1)
            scores = torch.abs(raw)
            recovered_mask = scores > float(threshold)
            n_recovered = int(recovered_mask.sum().item())
        else:
            raw = torch.empty(0)
            scores = torch.empty(0)
            recovered_mask = torch.empty(0, dtype=torch.bool)
            n_recovered = 0

        recovery_all = float(n_recovered / n_hidden) if n_hidden else np.nan
        recovery_eval = float(n_recovered / n_evaluable) if n_evaluable else np.nan

        # Pairing-null: shuffle only destination assignments among the SAME evaluable relation.
        null_eval_values = []
        null_all_values = []
        if n_evaluable >= 2 and int(null_permutations) > 0:
            src_t = torch.as_tensor(src_indices, dtype=torch.long)
            dst_np = np.asarray(dst_indices, dtype=np.int64)
            for _ in range(int(null_permutations)):
                perm_dst = torch.as_tensor(rng.permutation(dst_np), dtype=torch.long)
                null_scores = torch.abs(torch.sum(z[src_t] * z[perm_dst], dim=1))
                null_recovered = int((null_scores > float(threshold)).sum().item())
                null_eval_values.append(null_recovered / n_evaluable)
                null_all_values.append(null_recovered / n_hidden if n_hidden else np.nan)

        if null_all_values:
            null_all_arr = np.asarray(null_all_values, dtype=float)
            null_eval_arr = np.asarray(null_eval_values, dtype=float)
            null_all_mean = float(null_all_arr.mean())
            null_all_sd = float(null_all_arr.std(ddof=1)) if len(null_all_arr) > 1 else 0.0
            null_eval_mean = float(null_eval_arr.mean())
            null_eval_sd = float(null_eval_arr.std(ddof=1)) if len(null_eval_arr) > 1 else 0.0
            empirical_p = float((1 + np.sum(null_all_arr >= recovery_all)) / (len(null_all_arr) + 1))
            fold = float(recovery_all / null_all_mean) if null_all_mean > 0 else np.inf
            delta = float(recovery_all - null_all_mean)
        else:
            null_all_mean = null_all_sd = null_eval_mean = null_eval_sd = np.nan
            empirical_p = fold = delta = np.nan

        score_records = []
        for k, (edge, src_match, dst_match) in enumerate(evaluable_edges):
            score_records.append({
                'relation': relation,
                'src_type': str(edge.get('src_type')),
                'src_id': str(edge.get('src_id')),
                'dst_type': str(edge.get('dst_type')),
                'dst_id': str(edge.get('dst_id')),
                'src_match_mode': src_match,
                'dst_match_mode': dst_match,
                'raw_cosine': float(raw[k].item()),
                'score_abs_cosine': float(scores[k].item()),
                'threshold': float(threshold),
                'recovered': bool(recovered_mask[k].item()),
            })
        pd.DataFrame(score_records).to_csv(
            os.path.join(eval_dir, f'{relation}_recovery.csv'), index=False
        )

        metric_rows.append({
            'relation': relation,
            'seed': int(seed),
            'n_hidden': int(n_hidden),
            'n_evaluable': int(n_evaluable),
            'coverage': float(n_evaluable / n_hidden) if n_hidden else np.nan,
            'n_recovered': int(n_recovered),
            'missing_hidden_endpoints': int(relation_missing),
            'threshold': float(threshold),
            'RecoveryRecall': recovery_all,
            'RecoveryRecall_evaluable': recovery_eval,
            'NullRecoveryRecall_mean': null_all_mean,
            'NullRecoveryRecall_sd': null_all_sd,
            'NullRecoveryRecall_evaluable_mean': null_eval_mean,
            'NullRecoveryRecall_evaluable_sd': null_eval_sd,
            'RecoveryAboveNull': delta,
            'RecoveryFoldEnrichment': fold,
            'NullEmpiricalP': empirical_p,
            'null_permutations': int(null_permutations),
            'null_model': 'destination-permutation within relation (marginal-preserving pairing null)',
        })

        print(
            f" Held-out {relation}: hidden={n_hidden}, evaluable={n_evaluable}, "
            f"missing={relation_missing}, recovered={n_recovered}, "
            f"RecoveryRecall={recovery_all:.4f}, evaluable-only={recovery_eval:.4f}, "
            f"null={null_all_mean:.4f}±{null_all_sd:.4f}, p={empirical_p:.4g}"
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(eval_dir, 'metrics.csv'), index=False)

    if missing_records:
        missing_df = pd.DataFrame(missing_records)
        missing_df.to_csv(os.path.join(eval_dir, 'missing_hidden_edges.csv'), index=False)
        missing_df['reason'].value_counts().rename_axis('reason').reset_index(name='count').to_csv(
            os.path.join(eval_dir, 'missing_reason_counts.csv'), index=False
        )

    with open(os.path.join(eval_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'meta': hidden_meta,
            'seed': int(seed),
            'threshold': float(threshold),
            'null_permutations': int(null_permutations),
            'metric_definition': (
                'RecoveryRecall = recovered held-out edges / all held-out edges; '
                'pairing-null permutes destination endpoints within each relation.'
            ),
            'metrics': metric_rows,
        }, f, indent=2, ensure_ascii=False, default=str)
    return metrics_df


def write_run_all_metrics(
    experiment,
    seed,
    trainer,
    output_dir,
    alignment_summary,
    reconstruction_summary,
    heldout_metrics_df=None,
):
    """Write every reviewer-relevant scalar from ONE training seed into one tidy CSV."""
    rows = []

    def add(category, metric, value, relation='', direction='', notes=''):
        try:
            value_out = float(value) if value is not None and not pd.isna(value) else np.nan
        except Exception:
            value_out = np.nan
        rows.append({
            'experiment': str(experiment),
            'seed': int(seed),
            'category': category,
            'metric': metric,
            'relation_or_pair': relation,
            'value': value_out,
            'direction': direction,
            'notes': notes,
        })

    # Main manuscript integration metrics.
    add(
        'integration', 'Domain_accuracy_CV5',
        alignment_summary.get('domain_classifier_accuracy_cv5'),
        direction='closer to 0.333 is better for three modalities',
        notes='5 CV folds are averaged within this seed; folds are NOT independent experimental replicates.',
    )
    add('integration', 'iLISI_mean', alignment_summary.get('ilisi_mean'), direction='higher is better; max≈3')

    # Representation preservation / utilization.
    add('representation', 'Trustworthiness_weighted_mean', alignment_summary.get('trustworthiness_weighted_mean'), direction='higher is better')
    for mod, value in alignment_summary.get('trustworthiness_by_modality', {}).items():
        add('representation', f'Trustworthiness_{mod}', value, direction='higher is better')

    for metric, value in reconstruction_summary.items():
        add('representation', metric, value, direction='lower is better')

    # Held-out biological edge recovery

    if heldout_metrics_df is not None and len(heldout_metrics_df) > 0:
        for _, r in heldout_metrics_df.iterrows():
            rel = str(r['relation'])


            add(
                'heldout_edge_recovery',
                'RecoveryRecall',
                r.get('RecoveryRecall', np.nan),
                relation=rel,
                direction='higher is better',
                notes=(
                    'Fraction of held-out biological edges recovered '
                    'using the fixed |cosine similarity| threshold.'
                ),
            )

            add(
                'heldout_edge_recovery',
                'RecoveryRecall_evaluable',
                r.get('RecoveryRecall_evaluable', np.nan),
                relation=rel,
                direction='higher is better',
                notes='Recovery among held-out edges whose two endpoints have embeddings.',
            )
            add(
                'heldout_edge_recovery_null',
                'NullRecoveryRecall_mean',
                r.get('NullRecoveryRecall_mean', np.nan),
                relation=rel,
                direction='reference baseline',
                notes='Mean recovery under relation-wise destination permutation.',
            )
            add(
                'heldout_edge_recovery_null',
                'RecoveryFoldEnrichment',
                r.get('RecoveryFoldEnrichment', np.nan),
                relation=rel,
                direction='higher is better',
            )
            add(
                'heldout_edge_recovery_null',
                'NullEmpiricalP',
                r.get('NullEmpiricalP', np.nan),
                relation=rel,
                direction='lower is better',
            )


            add(
                'heldout_edge_recovery_meta',
                'n_hidden',
                r.get('n_hidden', np.nan),
                relation=rel,
                direction='count',
            )

            add(
                'heldout_edge_recovery_meta',
                'n_recovered',
                r.get('n_recovered', np.nan),
                relation=rel,
                direction='count',
            )

            add(
                'heldout_edge_recovery_meta',
                'n_evaluable',
                r.get('n_evaluable', np.nan),
                relation=rel,
                direction='count',
            )

            add(
                'heldout_edge_recovery_meta',
                'missing_hidden_endpoints',
                r.get('missing_hidden_endpoints', np.nan),
                relation=rel,
                direction='count',
            )

            add(
                'heldout_edge_recovery_meta',
                'coverage',
                r.get('coverage', np.nan),
                relation=rel,
                direction='higher is better',
            )

            add(
                'heldout_edge_recovery_meta',
                'threshold',
                r.get('threshold', np.nan),
                relation=rel,
                direction='fixed criterion',
            )

        # Training diagnostics are useful for reproducibility but are not manuscript performance claims.
        add('training_diagnostic', 'best_training_loss', trainer.best_loss, direction='diagnostic only')
        add('training_diagnostic', 'epochs_run', len(trainer.train_losses), direction='diagnostic only')

        out = pd.DataFrame(rows)
        path = os.path.join(output_dir, 'final_results', 'all_metrics.csv')
        out.to_csv(path, index=False)
        trainer.logger.info(f'All scalar metrics for this seed saved to: {path}')
        return out


def aggregate_all_metrics_results(output_root, experiment, seeds):
    """
    Aggregate run-level scalars across independent training seeds.

    The reported variability is SD across independent runs, not SD across CV folds/nodes.
    """
    frames = []
    missing = []
    for seed in seeds:
        path = os.path.join(
            output_root, experiment, f'seed_{seed}', 'final_results', 'all_metrics.csv'
        )
        if not os.path.exists(path):
            missing.append(path)
            continue
        df = pd.read_csv(path)
        df['seed'] = int(seed)
        frames.append(df)

    if missing:
        raise FileNotFoundError(
            'Cannot aggregate because some per-seed all_metrics.csv files are missing:\n' +
            '\n'.join(missing)
        )
    if not frames:
        raise RuntimeError('No all_metrics.csv files found.')

    all_df = pd.concat(frames, ignore_index=True)
    out_dir = os.path.join(output_root, experiment)
    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, 'all_metrics_all_seeds.csv')
    all_df.to_csv(all_path, index=False)

    group_cols = ['experiment', 'category', 'metric', 'relation_or_pair', 'direction', 'notes']
    summary_rows = []
    for keys, grp in all_df.groupby(group_cols, dropna=False, sort=True):
        vals = pd.to_numeric(grp['value'], errors='coerce').dropna().to_numpy(dtype=float)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row['n_runs'] = int(len(vals))
        if len(vals) > 0:
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row['mean'] = mean
            row['sd'] = sd
            row['mean±sd'] = f'{mean:.4f} ± {sd:.4f}'
        else:
            row['mean'] = np.nan
            row['sd'] = np.nan
            row['mean±sd'] = ''
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, 'all_metrics_mean_sd.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f' All metrics aggregated across {len(seeds)} seeds -> {summary_path}')
    return summary_df


def aggregate_heldout_results(output_root, experiment, seeds):
    """Aggregate held-out recovery and null-baseline metrics across independent seeds."""
    if experiment == 'full':
        raise ValueError('--aggregate is for cross_modal or tf_prior held-out experiments.')

    frames, missing = [], []
    for seed in seeds:
        path = os.path.join(
            output_root, experiment, f'seed_{seed}', 'final_results',
            'heldout_edge_recovery', 'metrics.csv'
        )
        if not os.path.exists(path):
            missing.append(path)
            continue
        df = pd.read_csv(path)
        df['seed'] = int(seed)
        frames.append(df)
    if missing:
        raise FileNotFoundError('Cannot aggregate because some seed results are missing:\n' + '\n'.join(missing))
    if not frames:
        raise RuntimeError('No held-out metric files found.')

    all_df = pd.concat(frames, ignore_index=True)
    rows = []
    metric_cols = [
        'RecoveryRecall', 'RecoveryRecall_evaluable', 'coverage',
        'NullRecoveryRecall_mean', 'NullRecoveryRecall_evaluable_mean',
        'RecoveryAboveNull', 'RecoveryFoldEnrichment', 'NullEmpiricalP',
    ]
    for relation, grp in all_df.groupby('relation', sort=True):
        thresholds = pd.to_numeric(grp['threshold'], errors='coerce').dropna().unique()
        if len(thresholds) != 1:
            raise RuntimeError(f'{relation}: inconsistent thresholds across seeds: {thresholds}')
        row = {'relation': relation, 'n_runs': int(len(grp)), 'threshold': float(thresholds[0])}
        for metric in metric_cols:
            if metric not in grp.columns:
                continue
            vals = pd.to_numeric(grp[metric], errors='coerce').dropna()
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f'{metric}_mean'] = mean
            row[f'{metric}_sd'] = sd
            row[f'{metric}_mean±sd'] = f'{mean:.4f} ± {sd:.4f}'
        row['hidden_edges_mean'] = float(pd.to_numeric(grp['n_hidden'], errors='coerce').mean())
        row['recovered_edges_mean'] = float(pd.to_numeric(grp['n_recovered'], errors='coerce').mean())
        row['missing_endpoints_mean'] = float(pd.to_numeric(grp['missing_hidden_endpoints'], errors='coerce').mean())
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_dir = os.path.join(output_root, experiment)
    os.makedirs(out_dir, exist_ok=True)
    all_df.to_csv(os.path.join(out_dir, 'heldout_recovery_all_seeds.csv'), index=False)
    summary_path = os.path.join(out_dir, 'heldout_recovery_mean_sd.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Aggregated {len(seeds)} seeds -> {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df

# Entry point

def main():
    parser = argparse.ArgumentParser(
        description='PoplarCrossOmics training + reviewer held-out edge validation'
    )
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID (use -1 for CPU)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1, help='Experiment seed; must match mask seed')
    parser.add_argument(
        '--experiment',
        choices=['full', 'cross_modal', 'tf_prior'],
        default='full',
        help='full=original graph; cross_modal/tf_prior=held-out reviewer validation',
    )
    parser.add_argument('--graph-root', default=DEFAULT_GRAPH_ROOT)
    parser.add_argument('--heldout-root', default=DEFAULT_HELDOUT_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
    '--recovery-threshold',
    type=float,
    default=0.5,
    help='Absolute cosine-similarity threshold for held-out edge recovery.',
    )
    parser.add_argument(
        '--null-permutations',
        type=int,
        default=100,
        help='Number of relation-wise destination permutations for the recovery null baseline.',
    )
    parser.add_argument(
        '--knn-k',
        type=int,
        default=30,
        help='Neighborhood size for iLISI and trustworthiness calculations.',
    )
    parser.add_argument(
        '--trust-max-samples',
        type=int,
        default=None,
        help='Optional per-modality cap for trustworthiness computation; default uses all nodes.',
    )
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Do not train; aggregate all reviewer-relevant metrics across --aggregate-seeds.',
    )
    parser.add_argument(
        '--aggregate-seeds',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
    )
    args = parser.parse_args()

    if args.aggregate:
        aggregate_all_metrics_results(args.output_root, args.experiment, args.aggregate_seeds)
        if args.experiment != 'full':
            aggregate_heldout_results(args.output_root, args.experiment, args.aggregate_seeds)
        return

    if not 0.0 <= args.recovery_threshold <= 1.0:
        raise ValueError(
            '--recovery-threshold must be between 0 and 1.'
        )
    if args.null_permutations < 0:
        raise ValueError('--null-permutations must be >= 0.')
    if args.knn_k < 1:
        raise ValueError('--knn-k must be >= 1.')
    if args.trust_max_samples is not None and args.trust_max_samples < 4:
        raise ValueError('--trust-max-samples must be >= 4 when supplied.')

    graph_path, hidden_path, manifest_path, output_dir = resolve_experiment_paths(args)
    manifest = validate_experiment_binding(
        experiment=args.experiment,
        seed=args.seed,
        graph_path=graph_path,
        hidden_path=hidden_path,
        manifest_path=manifest_path,
    )

    # Set RNG BEFORE data/model construction. The same experiment seed controls training.
    set_global_seed(args.seed)

    os.makedirs(output_dir, exist_ok=True)
    print('=' * 88)
    print(f'Experiment       : {args.experiment}')
    print(f'Experiment seed  : {args.seed}')
    print(f'Training graph   : {graph_path}')
    print(f'Hidden test edges: {hidden_path if hidden_path else "N/A (full graph)"}')
    print(f'Output directory : {output_dir}')
    print('=' * 88)

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )
    print(f'Using device: {device}')

    data_paths = {
        'graph': graph_path,
        'rna': r"/home/nefu1000004427/cwh/1/pyHGT-master/process/filtered_omics_genes/rna_common_genes_matrix.csv",
        'methylation': r"/home/nefu1000004427/cwh/1/pyHGT-master/process/filtered_omics_genes/methylation_common_genes_matrix.csv",
        'snp': r"/home/nefu1000004427/cwh/1/pyHGT-master/process/filtered_omics_genes/snp_common_genes_matrix.csv",
    }

    print('\nChecking data file existence...')
    missing = [f'{name}: {path}' for name, path in data_paths.items() if not os.path.exists(path)]
    if missing:
        print(' Missing required files:')
        print('\n'.join(missing))
        sys.exit(1)
    for name, path in data_paths.items():
        print(f' Found {name}: {path}')

    try:
        print('\nInitializing config and data loader...')
        model_config = ModelConfig()
        model_config.num_epochs = args.epochs
        model_config.learning_rate = args.lr
        model_config.random_seed = int(args.seed)

        data_loader = MultiOmicsDataLoader(data_paths)
        data_loader.set_seed(args.seed)
        data = data_loader.load_data()

        # Before any training, re-check that held-out positives are absent from the graph.
        hidden_by_relation = None
        if hidden_path is not None:
            _, hidden_by_relation = load_hidden_payload(hidden_path)
            verify_no_hidden_edge_leakage(data, hidden_by_relation)
            print(' Leakage check passed: all held-out positives are absent from training graph.')

        for omics_type, tensor in data['omics_data_dict'].items():
            if tensor.numel() > 0:
                actual_dim = tensor.shape[1]
                model_config.source_params[omics_type]['input_dim'] = actual_dim
                print(f'Updated {omics_type} input_dim to {actual_dim} (matches actual data)')

        data = preprocess_omics_data(data)

        trainer = GraphTrainer(model_config, device, output_dir)
        trainer.build_model(data)

        run_manifest = {
            'experiment': args.experiment,
            'seed': int(args.seed),
            'graph_path': os.path.abspath(graph_path),
            'hidden_edge_path': os.path.abspath(hidden_path) if hidden_path else None,
            'output_dir': os.path.abspath(output_dir),
            'mask_manifest': manifest,

            'heldout_recovery_threshold': float(
                args.recovery_threshold
            ),
            'heldout_recovery_metric': 'RecoveryRecall',
            'heldout_null_permutations': int(args.null_permutations),
            'heldout_null_model': 'destination-permutation within relation',
            'knn_k': int(args.knn_k),
            'trust_max_samples': args.trust_max_samples,
        }
        with open(os.path.join(output_dir, 'run_manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(run_manifest, f, indent=2, ensure_ascii=False, default=str)

        print('\nStarting training...')
        _ = trainer.train(data)

        print('\nSaving final results...')
        node_out, alignment_summary, reconstruction_summary = save_final_results(
            trainer, data, output_dir, eval_seed=args.seed,
            knn_k=args.knn_k, trust_max_samples=args.trust_max_samples,
        )

        heldout_metrics_df = None
        if hidden_path is not None:
            print('\nEvaluating held-out biological edges...')
            heldout_metrics_df = evaluate_hidden_edges(
                node_out=node_out,
                data=data,
                hidden_path=hidden_path,
                output_dir=output_dir,
                seed=args.seed,
                threshold=args.recovery_threshold,
                null_permutations=args.null_permutations,
            )


        write_run_all_metrics(
            experiment=args.experiment,
            seed=args.seed,
            trainer=trainer,
            output_dir=output_dir,
            alignment_summary=alignment_summary,
            reconstruction_summary=reconstruction_summary,
            heldout_metrics_df=heldout_metrics_df,
        )

        # Reuse the exact same node_out; do not perform another stochastic VAE sampling.
        create_visualizations(
            trainer, data, output_dir, node_out=node_out, eval_seed=args.seed
        )

        print('\n' + '=' * 88)
        print('Training Summary')
        print('=' * 88)
        print(f'Experiment: {args.experiment}')
        print(f'Seed: {args.seed}')
        print(f'Best Training Loss: {trainer.best_loss:.6f}')
        print(f'Total Epochs Run: {len(trainer.train_losses)}')
        print(f'Final Learning Rate: {trainer.optimizer.param_groups[0]["lr"]:.6e}')
        print(f'All Results Saved To: {os.path.abspath(output_dir)}')
        if hidden_path is not None:
            print('Held-out metrics: final_results/heldout_edge_recovery/metrics.csv')
        print('=' * 88)

    except Exception as e:
        print(f'\n Training failed with error: {e}')
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
