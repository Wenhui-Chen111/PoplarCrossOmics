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
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gc
from model2 import *

def clean_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  

BASE_NODE_TYPE_MAPPING = {'SNP': 0, 'METH': 1, 'RNA': 2}
DATA_NODE_TYPE_MAPPING = {'SNP': 0, 'METH': 1, 'RNA': 2, 'unknown': 3}

class ModelConfig:
    def __init__(self):
      
        self.source_params = {
            'rna': {'input_dim': None, 'hidden_dims': [128, 64], 'latent_dim': 32},
            'methylation': {'input_dim': None, 'hidden_dims': [128, 64], 'latent_dim': 32},
            'snp': {'input_dim': None, 'hidden_dims': [128, 64], 'latent_dim': 32}
        }
        self.node_type_mapping = BASE_NODE_TYPE_MAPPING.copy()

        self.n_hid = 64
        self.n_heads = 4
        self.hgt_layers = 2
        self.dropout = 0.2
        self.use_norm = True
    
        self.adv_weight = 0.1  
        self.mmd_weight = 0.1
        self.adv_lambda = 1.0    

        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 100
        self.early_stopping_patience = 30

        self.kl_beta_start = 0.0
        self.kl_beta_end = 0.1
        self.kl_warmup_epochs = 300

        self.log_detailed_metrics_every = 1
        self.log_latent_features_every = 300



class DataConfig:
    def __init__(self):
        self.node_type_mapping = DATA_NODE_TYPE_MAPPING.copy()


class MultiOmicsDataLoader:
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.config = DataConfig()

    def set_seed(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

   
    def load_omics_features(self):
        features = {}
        for feature_type in ['rna', 'methylation', 'snp']:
            df = pd.read_csv(self.data_paths[feature_type], index_col=0)
            df.index = df.index.astype(str)
            features[feature_type] = df
        return features

    def create_node_mappings(self, graph_data, omics_features, add_orphan_nodes=False):
 
        nodes_by_type = {}

        graph_type_to_ids = {'RNA': [], 'METH': [], 'SNP': []}
        for t, i in graph_data['nodes']:
            if t in graph_type_to_ids:
                graph_type_to_ids[t].append(str(i))

        csv_ids = {
            'RNA':  [str(idx) for idx in omics_features['rna'].index.tolist()],
            'METH': [str(idx) for idx in omics_features['methylation'].index.tolist()],
            'SNP':  [str(idx) for idx in omics_features['snp'].index.tolist()],
        }

        def uniq_keep_order(seq):
            seen = set(); out = []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out

        for nt in ['SNP', 'METH', 'RNA']:
            order = list(csv_ids[nt]) 
            if add_orphan_nodes:
                extras = [gid for gid in graph_type_to_ids[nt] if gid not in csv_ids[nt]]
                order.extend(extras)
            order = uniq_keep_order(order)
            nodes_by_type[nt] = [(nt, nid) for nid in order]

        type_idx_mappings = {nt: {node: idx for idx, node in enumerate(lst)} for nt, lst in nodes_by_type.items()}
        return {'type_idx_mappings': type_idx_mappings, 'nodes_by_type': nodes_by_type}



    @staticmethod
    def _clean_id(id_str):
        if isinstance(id_str, str):
            return id_str.strip().lower().replace('-', '').replace('_', '').replace('(', '').replace(')', '')
        return str(id_str)

    def create_omics_data_dict(self, omics_features, mappings, fill_missing="zeros"):
        
        import numpy as np

        omics_to_node = {'rna': 'RNA', 'methylation': 'METH', 'snp': 'SNP'}
        omics_data_dict = {}

        for omics_type, node_type in omics_to_node.items():
            nodes = mappings['nodes_by_type'].get(node_type, [])
            if not nodes:
                omics_data_dict[omics_type] = torch.empty(0, 1, device=self.get_device())
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
                    else:  
                        feats.append(col_mean)

            arr = np.stack(feats, axis=0)
            omics_data_dict[omics_type] = torch.from_numpy(arr)
        return omics_data_dict

    def parse_graph_data(self, data_dict):
        print("Parsing saved graph data dictionary...")
        if 'edge_info_list' not in data_dict or len(data_dict['edge_info_list']) == 0:
            raise ValueError("edge_info_list not found or empty in saved data")
        edge_info_list = data_dict['edge_info_list']

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

        all_nodes = list({(e['src_type'], e['src_id']) for e in edges_info} | {(e['dst_type'], e['dst_id']) for e in edges_info})

        def _count(nodes):
            d = {}
            for t, _ in nodes:
                d[t] = d.get(t, 0) + 1
            return d

        return {
            'nodes': all_nodes, 'edges_info': edges_info,
            'edge_weights': [e['weight'] for e in edges_info],
            'relations': list(set(e['relation'] for e in edges_info)),
            'node_type_counts': _count(all_nodes)
        }

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

        found_relations = set(e['relation'] for e in graph_data['edges_info'])
        edge_type_mapping = {rel: idx for idx, rel in enumerate(found_relations)}

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

    def load_data(self):
        print("Loading multi-omics data...")
        final = {}

        print("Step 1/4: Loading graph structure...")
        with open(self.data_paths['graph'], 'rb') as f:
            saved = pickle.load(f)
        raw_edge_info = saved['edge_info_list'] if isinstance(saved, dict) and 'edge_info_list' in saved else None
        if raw_edge_info is None:
            raise ValueError("Invalid graph data structure: missing 'edge_info_list'")
        graph_data = self.parse_graph_data({'edge_info_list': raw_edge_info})
        final['graph_data'] = graph_data

        print("\nStep 2/4: Loading omics features...")
        omics_features = self.load_omics_features()
        final['omics_features'] = omics_features

        print("\nStep 3/4: Creating node mappings...")
        mappings = self.create_node_mappings(graph_data, omics_features, add_orphan_nodes=False)

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
            hgt_layers=self.config.hgt_layers
        ).to(self.device)
   
        self.model.kl_beta_start = self.config.kl_beta_start
        self.model.kl_beta_end = self.config.kl_beta_end
        self.model.kl_warmup_epochs = self.config.kl_warmup_epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total model parameters: {total_params:,}")

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

        latent_stats = {}
        if (epoch + 1) % self.config.log_latent_features_every == 0:
            node_emb = model_out['node_emb']
            latent_dict_for_stats = {
                'rna': node_emb[node_type_tensor == self.config.node_type_mapping['RNA']],
                'methylation': node_emb[node_type_tensor == self.config.node_type_mapping['METH']],
                'snp': node_emb[node_type_tensor == self.config.node_type_mapping['SNP']],
            }
            del node_emb
            
        torch.cuda.empty_cache()
        clean_cuda_cache()
        return total_loss.item(), loss_dict, {'latent_stats': latent_stats}

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
            'detailed_metrics': self.detailed_metrics,
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
        plt.xlabel('Epoch'); plt.ylabel('Loss (log-scale)')
        plt.title('Training losses (log scale)')
        plt.grid(alpha=0.3); plt.legend(ncol=3)
        plt.tight_layout()
        p1 = os.path.join(stats_dir, "loss_logscale.png")
        plt.savefig(p1, dpi=300, bbox_inches='tight'); plt.close()
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
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (left)')
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
            plt.savefig(p2, dpi=300, bbox_inches='tight'); plt.close()
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

def preprocess_omics_data(data):
    rna_raw = data['omics_data_dict']['rna']
    if rna_raw.numel() > 0:
        rna_std = standardize(torch.log1p(rna_raw), dim=1, clip_extreme=False)
        data['omics_data_dict']['rna'] = rna_std
        glob_var = float(rna_std.var())

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

    snp_raw = data['omics_data_dict']['snp']
    if snp_raw.numel() > 0:
        snp_std = standardize(torch.log1p(snp_raw), dim=1, clip_extreme=False)
        data['omics_data_dict']['snp'] = snp_std
        glob_var = float(snp_std.var())

    return data

@torch.no_grad()
def save_final_results(trainer, data, output_dir):
    trainer.model.eval()
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

    compute_alignment_metrics(trainer, output_dir, which="node_out")


def compute_alignment_metrics(trainer,output_dir, which="node_out", knn_k=30):

    results_dir = os.path.join(output_dir, "final_results", "alignment_metrics", which)
    os.makedirs(results_dir, exist_ok=True)

    base_dir_map = {
        "node_out":   os.path.join(output_dir, "final_results", "node_embeddings"),
        "node_emb":   os.path.join(output_dir, "final_results", "raw_node_embeddings"),
        "vae_latent": os.path.join(output_dir, "final_results", "vae_latent_embeddings"),
    }
    bdir = base_dir_map[which]

    def read_type(tname):
        p = os.path.join(bdir, f"{tname.lower()}_{which}_embeddings.csv")
        if not os.path.exists(p): return None
        df = pd.read_csv(p)
        z = df.filter(regex=fr"^{which}_dim_").to_numpy()
        ids = df["node_id"].astype(str).to_numpy()
        return ids, z

    r = read_type("RNA"); m = read_type("METH"); s = read_type("SNP")
    assert r is not None, "RNA embeddings missing"

    ids_all, Z_all, labels = [], [], []
    for name, tup in [("RNA", r), ("METH", m), ("SNP", s)]:
        if tup is None: continue
        ids, Z = tup
        ids_all.append(ids); Z_all.append(Z); labels += [name]*len(ids)
    Z_all = np.vstack(Z_all); labels = np.array(labels)
    kk = min(knn_k, max(2, len(labels)-1))
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine").fit(Z_all)
    _, idx = nn.kneighbors(Z_all, return_distance=True)
    mods = np.unique(labels)
    ilisi = []
    for i in range(Z_all.shape[0]):
        neigh = labels[idx[i]]
        pm = np.array([(neigh==m).mean() for m in mods])
        ilisi.append(1.0/np.sum(pm**2))
    ilisi = np.array(ilisi)
    pd.DataFrame({"node_index":np.arange(len(ilisi)),"iLISI":ilisi,"modality":labels}).to_csv(
        os.path.join(results_dir,"ilisi_per_node.csv"), index=False)

    if len(np.unique(labels)) >= 2:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        for tr, te in skf.split(Z_all, labels):
            clf.fit(Z_all[tr], labels[tr])
            accs.append(accuracy_score(labels[te], clf.predict(Z_all[te])))
        domain_acc = float(np.mean(accs))
    else:
        domain_acc = float('nan')

    summary = {
        "which_embedding": which,
        "ilisi_mean": float(np.nanmean(ilisi)) if len(ilisi)>0 else float('nan'),
        "ilisi_max_possible": int(len(mods)),
        "domain_classifier_accuracy_cv5": domain_acc
    }
    with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    trainer.logger.info(f"Alignment metrics saved to {results_dir}")

def main():
    parser = argparse.ArgumentParser(description='PoplarCrossOmicsModel Training (adapted)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (use -1 for CPU)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibitestlity')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to graph pickle file')
    parser.add_argument('--rna_path', type=str, required=True, help='Path to RNA csv file')
    parser.add_argument('--methylation_path', type=str, required=True, help='Path to methylation csv file')
    parser.add_argument('--snp_path', type=str, required=True, help='Path to SNP csv file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device: {device}")

    data_paths = {
    'graph': args.graph_path,
    'rna': args.rna_path,
    'methylation': args.methylation_path,
    'snp': args.snp_path
    }

    print("\nChecking data file existence...")
    all_exist = True
    for name, path in data_paths.items():
        if not os.path.exists(path):
            print(f" Missing {name} file: {path}")
            all_exist = False
        else:
            print(f" Found {name} file: {path}")
    if not all_exist:
        print("Please correct data paths and retry.")
        sys.exit(1)

    try:
        print("\nInitializing config and data loader...")
        model_config = ModelConfig()
        model_config.num_epochs = args.epochs
        model_config.learning_rate = args.lr

        data_loader = MultiOmicsDataLoader(data_paths)
        data_loader.set_seed(args.seed)
        data = data_loader.load_data()

        for omics_type, tensor in data['omics_data_dict'].items():
            if tensor.numel() > 0:
                actual_dim = tensor.shape[1]
                model_config.source_params[omics_type]['input_dim'] = actual_dim

        data = preprocess_omics_data(data)

        trainer = GraphTrainer(model_config, device, output_dir)
        trainer.build_model(data)

        print("\nStarting training...")
        _ = trainer.train(data)

        print("\nSaving final results...")
        save_final_results(trainer, data, output_dir)

    except Exception as e:
        print(f"\n Training failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
