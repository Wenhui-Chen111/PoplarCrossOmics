# -*- coding: utf-8 -*-
# ============================================================
# run2.py — PoplarCrossOmicsModel Training Script
# Purpose: Train the model with VAE, Adversarial, and MMD losses.
#          Removes contrastive/ID parts and complex visualizations.
# ============================================================

import os
# Optimize CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import sys
import traceback
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import json
import gc

# Import the modified model
from model2 import PoplarCrossOmicsModel

def clean_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()

# ============================== Configuration ==============================

class ModelConfig:
    def __init__(self):
        # Omics encoder parameters (input_dim updated dynamically)
        self.source_params = {
            'rna': {'input_dim': 2000, 'hidden_dims': [128, 64], 'latent_dim': 32},
            'methylation': {'input_dim': 1500, 'hidden_dims': [128, 64], 'latent_dim': 32},
            'snp': {'input_dim': 1000, 'hidden_dims': [128, 64], 'latent_dim': 32}
        }
        self.node_type_mapping = {'SNP': 0, 'METH': 1, 'RNA': 2}

        # Model Architecture
        self.n_hid = 64
        self.n_heads = 4
        self.hgt_layers = 2
        self.dropout = 0.2
        self.use_norm = True

        # Loss Weights (Contrastive and ID removed)
        self.adv_weight = 0.05   # Adversarial loss weight
        self.mmd_weight = 0.05   # MMD loss weight
        self.adv_lambda = 1.0    # GRL lambda

        # Training Hyperparameters
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 100
        self.early_stopping_patience = 30

        # KL Annealing
        self.kl_beta_start = 0.0
        self.kl_beta_end = 0.01
        self.kl_warmup_epochs = 300

        # Logging
        self.log_detailed_metrics_every = 1

class DataConfig:
    def __init__(self):
        self.node_type_mapping = {'SNP': 0, 'METH': 1, 'RNA': 2, 'unknown': 3}

# ============================== Data Loader ==============================

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

    def parse_graph_data(self, data_dict):
        print("Parsing saved graph data...")
        if 'edge_info_list' not in data_dict:
            raise ValueError("edge_info_list not found")
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

        all_nodes = list({(e['src_type'], e['src_id']) for e in edges_info} | 
                         {(e['dst_type'], e['dst_id']) for e in edges_info})
        
        return {
            'nodes': all_nodes, 
            'edges_info': edges_info,
            'relations': list(set(e['relation'] for e in edges_info))
        }

    def load_omics_features(self):
        print("Loading omics features...")
        features = {}
        for feature_type in ['rna', 'methylation', 'snp']:
            df = pd.read_csv(self.data_paths[feature_type], index_col=0)
            df.index = df.index.astype(str)
            features[feature_type] = df
            print(f"{feature_type.upper()} shape: {df.shape}")
        return features

    def create_node_mappings(self, graph_data, omics_features):
        print("Creating node mappings...")
        nodes_by_type = {}
        
        # Get IDs from CSVs
        csv_ids = {
            'RNA':  [str(idx) for idx in omics_features['rna'].index.tolist()],
            'METH': [str(idx) for idx in omics_features['methylation'].index.tolist()],
            'SNP':  [str(idx) for idx in omics_features['snp'].index.tolist()],
        }

        # Unique nodes keeping CSV order
        for nt in ['SNP', 'METH', 'RNA']:
            unique_ids = []
            seen = set()
            for x in csv_ids[nt]:
                if x not in seen:
                    seen.add(x)
                    unique_ids.append(x)
            nodes_by_type[nt] = [(nt, nid) for nid in unique_ids]

        type_idx_mappings = {nt: {node: idx for idx, node in enumerate(lst)} 
                             for nt, lst in nodes_by_type.items()}
        return {'type_idx_mappings': type_idx_mappings, 'nodes_by_type': nodes_by_type}

    @staticmethod
    def _clean_id(id_str):
        if isinstance(id_str, str):
            return id_str.strip().lower().replace('-', '').replace('_', '')
        return str(id_str)

    def create_omics_data_dict(self, omics_features, mappings):
        print("Creating omics data tensors...")
        omics_to_node = {'rna': 'RNA', 'methylation': 'METH', 'snp': 'SNP'}
        omics_data_dict = {}

        for omics_type, node_type in omics_to_node.items():
            nodes = mappings['nodes_by_type'].get(node_type, [])
            if not nodes:
                omics_data_dict[omics_type] = torch.empty(0, 1)
                continue

            node_ids = [nid for _, nid in nodes]
            omics_df = omics_features[omics_type]
            col_mean = omics_df.mean(axis=0).values.astype(np.float32)
            
            # Map clean IDs for fast lookup
            pos = {self._clean_id(str(idx)): i for i, idx in enumerate(omics_df.index.tolist())}

            feats = []
            for nid in node_ids:
                j = pos.get(self._clean_id(str(nid)))
                if j is not None:
                    feats.append(omics_df.iloc[j].values.astype(np.float32))
                else:
                    feats.append(col_mean) # Fill missing with mean

            arr = np.stack(feats, axis=0)
            omics_data_dict[omics_type] = torch.from_numpy(arr)
        
        return omics_data_dict

    def process_edges(self, graph_data, mappings):
        print("Processing edges...")
        from collections import defaultdict
        
        # Create global index map
        node_to_global_idx = {}
        cur = 0
        all_nodes_ordered = []
        for nt in ['SNP', 'METH', 'RNA']:
            for node in mappings['nodes_by_type'].get(nt, []):
                all_nodes_ordered.append(node)
                node_to_global_idx[node] = cur
                cur += 1
        
        found_relations = set(e['relation'] for e in graph_data['edges_info'])
        edge_type_mapping = {rel: idx for idx, rel in enumerate(found_relations)}

        edge_indices_dict = defaultdict(list)
        edge_weights_dict = defaultdict(list)
        edge_signs_dict = defaultdict(list)

        for e in graph_data['edges_info']:
            src = (e['src_type'], e['src_id'])
            dst = (e['dst_type'], e['dst_id'])
            rel = e['relation']
            raw_sign = e.get('sign', 1)
            weight = float(e.get('weight', 1.0))

            if src not in node_to_global_idx or dst not in node_to_global_idx:
                continue
            
            # Normalize sign: 0 or < -1 -> -2 (unknown), >1 -> 1
            if raw_sign == 0 or raw_sign < -1:
                sign = -2
            elif raw_sign > 1:
                sign = 1
            else:
                sign = int(raw_sign)

            src_idx = node_to_global_idx[src]
            dst_idx = node_to_global_idx[dst]
            
            edge_indices_dict[rel].append([src_idx, dst_idx])
            edge_weights_dict[rel].append(weight)
            edge_signs_dict[rel].append(sign)

        # Convert to tensors
        for rel in edge_indices_dict:
            edge_indices_dict[rel] = torch.LongTensor(edge_indices_dict[rel]).T
            edge_weights_dict[rel] = torch.FloatTensor(edge_weights_dict[rel])
            edge_signs_dict[rel] = torch.LongTensor(edge_signs_dict[rel])

        # Handle empty relations
        for rel in found_relations:
            if rel not in edge_indices_dict:
                edge_indices_dict[rel] = torch.empty((2, 0), dtype=torch.long)
                edge_weights_dict[rel] = torch.empty(0, dtype=torch.float)
                edge_signs_dict[rel] = torch.empty(0, dtype=torch.long)

        return {
            'edge_indices_dict': edge_indices_dict,
            'edge_weights_dict': edge_weights_dict,
            'edge_signs_dict': edge_signs_dict,
            'edge_type_mapping': edge_type_mapping,
            'all_nodes_ordered': all_nodes_ordered,
            'valid_relations': list(found_relations)
        }

    def load_data(self):
        print("Starting data load...")
        final = {}
        
        # 1. Graph Structure
        with open(self.data_paths['graph'], 'rb') as f:
            saved = pickle.load(f)
        graph_data = self.parse_graph_data(saved)
        
        # 2. Features
        omics_features = self.load_omics_features()
        
        # 3. Mappings
        mappings = self.create_node_mappings(graph_data, omics_features)
        final.update(mappings)
        
        # 4. Tensors
        final['omics_data_dict'] = self.create_omics_data_dict(omics_features, mappings)
        
        # 5. Edges & Types
        edge_data = self.process_edges(graph_data, mappings)
        final.update(edge_data)

        node_type_list = [self.config.node_type_mapping.get(n[0], 3) for n in final['all_nodes_ordered']]
        final['node_type_tensor'] = torch.tensor(node_type_list, dtype=torch.long)
        
        return final

# ============================== Trainer ==============================

class GraphTrainer:
    def __init__(self, config, device, output_dir):
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        
        # Metrics storage (Cleaned up)
        self.detailed_metrics = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': [], 
            'mmd_loss': [], 'adv_loss': [], 'kl_beta': []
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
        self.logger.info(f"Log started: {log_file}")

    def build_model(self, data):
        self.logger.info("Building Model...")
        self.model = PoplarCrossOmicsModel(
            source_params=self.config.source_params,
            node_type_mapping=self.config.node_type_mapping,
            edge_type_mapping=data['edge_type_mapping'],
            n_hid=self.config.n_hid,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            use_norm=self.config.use_norm,
            adv_weight=self.config.adv_weight,
            mmd_weight=self.config.mmd_weight,
            adv_lambda=self.config.adv_lambda,
            hgt_layers=self.config.hgt_layers
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

    def train_epoch(self, data, epoch):
        clean_cuda_cache()
        self.model.train()
        device = self.device

        # Move data to device
        omics_data = {k: v.to(device, non_blocking=True) for k, v in data['omics_data_dict'].items()}
        edge_indices = {k: v.to(device, non_blocking=True) for k, v in data['edge_indices_dict'].items()}
        edge_weights = {k: v.to(device, non_blocking=True) for k, v in data['edge_weights_dict'].items()}
        edge_signs = {k: v.to(device, non_blocking=True).long() for k, v in data['edge_signs_dict'].items()}
        node_types = data['node_type_tensor'].to(device, non_blocking=True)

        # Ensure signs match dimensions
        for rel, eidx in edge_indices.items():
            if edge_signs.get(rel, torch.empty(0)).numel() != eidx.size(1):
                edge_signs[rel] = torch.full((eidx.size(1),), -2, dtype=torch.long, device=device)

        self.optimizer.zero_grad()
        
        # Forward Pass
        model_out = self.model(
            omics_data_dict=omics_data,
            edge_indices_dict=edge_indices,
            edge_weights_dict=edge_weights,
            edge_signs_dict=edge_signs,
            node_type_tensor=node_types,
            epoch=epoch
        )

        loss_dict = model_out['loss_dict']
        total_loss = loss_dict['total_loss']
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Logging
        if (epoch + 1) % self.config.log_detailed_metrics_every == 0:
            self.detailed_metrics['total_loss'].append(total_loss.item())
            self.detailed_metrics['recon_loss'].append(loss_dict['reconstruction_loss'].item())
            self.detailed_metrics['kl_loss'].append(loss_dict['kl_loss'].item())
            self.detailed_metrics['mmd_loss'].append(loss_dict['mmd_loss'].item())
            self.detailed_metrics['adv_loss'].append(loss_dict['adv_loss'].item())
            self.detailed_metrics['kl_beta'].append(self.model.compute_kl_beta(epoch))

            self.logger.info(f"Epoch {epoch+1} | Loss: {total_loss.item():.4f} | "
                             f"Recon: {loss_dict['reconstruction_loss'].item():.4f} | "
                             f"MMD: {loss_dict['mmd_loss'].item():.4f} | "
                             f"Adv: {loss_dict['adv_loss'].item():.4f}")

        return total_loss.item()

    def train(self, data):
        self.logger.info("Starting Training...")
        pbar = tqdm(range(self.config.num_epochs))
        
        for epoch in pbar:
            loss = self.train_epoch(data, epoch)
            self.train_losses.append(loss)
            self.scheduler.step(loss)
            
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
                self.save_model(os.path.join(self.output_dir, 'best_model.pth'))
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered.")
                    break
            
            pbar.set_description(f"Loss: {loss:.4f}")

        self.plot_training_curve()
        self.logger.info("Training Completed.")

    def save_model(self, path):
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }
        torch.save(ckpt, path)

    def plot_training_curve(self):
        if not self.train_losses: return
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'training_curve.png'))
        plt.close()

# ============================== Preprocessing ==============================

def preprocess_omics_data(data):
    print("Preprocessing data (Normalization)...")
    
    def standardize(tensor, dim, clip_percentile=None):
        if tensor.numel() == 0: return tensor
        x = tensor.clone()
        if clip_percentile:
            lower = torch.quantile(x, clip_percentile[0]/100.0, dim=dim, keepdim=True)
            upper = torch.quantile(x, clip_percentile[1]/100.0, dim=dim, keepdim=True)
            x = torch.clamp(x, lower, upper)
        
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return torch.nan_to_num((x - mean) / std)

    # RNA & SNP: Log1p + Row Norm
    for k in ['rna', 'snp']:
        raw = data['omics_data_dict'][k]
        if raw.numel() > 0:
            data['omics_data_dict'][k] = standardize(torch.log1p(raw), dim=1)

    # Methylation: Logit + Clip + Col Norm
    meth = data['omics_data_dict']['methylation']
    if meth.numel() > 0:
        x = meth.clone()
        if x.max() > 1.05: x = x / 100.0
        x = torch.clamp(x, 1e-6, 1-1e-6)
        x = torch.logit(x)
        data['omics_data_dict']['methylation'] = standardize(x, dim=0, clip_percentile=(1, 99))

    return data

# ============================== Main ==============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # Paths
    output_dir = "./result_output" # Modified relative path for safety
    os.makedirs(output_dir, exist_ok=True)
    
    # Update these paths to your actual file locations
    data_paths = {
        'graph': r"/home/nefu1000004427/cwh/pyHGT-master/graph/guide_graph_pruned.pkl",
        'rna': r"/home/nefu1000004427/cwh/pyHGT-master/process/filtered_omics_genes/rna_common_genes_matrix.csv",
        'methylation': r"/home/nefu1000004427/cwh/pyHGT-master/process/filtered_omics_genes/methylation_common_genes_matrix.csv",
        'snp': r"/home/nefu1000004427/cwh/pyHGT-master/process/filtered_omics_genes/snp_common_genes_matrix.csv"
    }

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    try:
        # Load & Process
        loader = MultiOmicsDataLoader(data_paths)
        data = loader.load_data()
        data = preprocess_omics_data(data)
        
        # Config Update
        config = ModelConfig()
        config.num_epochs = args.epochs
        config.learning_rate = args.lr
        for k, v in data['omics_data_dict'].items():
            if v.numel() > 0: config.source_params[k]['input_dim'] = v.shape[1]

        # Train
        trainer = GraphTrainer(config, device, output_dir)
        trainer.build_model(data)
        trainer.train(data)

        # Final Save (Simplified)
        print("Saving final results...")
        torch.save(trainer.model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        
        # Save training log to CSV
        df = pd.DataFrame(trainer.detailed_metrics)
        df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)

        print(f"Done. Results saved to {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
