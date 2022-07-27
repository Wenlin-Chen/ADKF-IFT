from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn
import numpy as np

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
from fs_mol.data.dkt import DKTBatch

from fs_mol.utils.gp_utils import ExactGPLayer

import gpytorch
from gpytorch.distributions import MultivariateNormal

#from fs_mol.utils._stateless import functional_call

import sys
sys.path.append("./PAR-NeurIPS21/")
from chem_lib.models.mol_model import attention, ContextAwareRelationNet
from chem_lib.models.relation import MLP, ContextMLP, TaskAwareRelation

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class PARModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    #distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class PARModel(nn.Module):
    def __init__(self, config: PARModelConfig):
        super().__init__()
        self.config = config

        # TODO: replace "args" with correct params in "config"
        # TODO: update config to contain necessary params

        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                config.graph_feature_extractor_config
            )

        self.use_fc = self.config.used_features.endswith("+fc")

        # Create MLP if needed:
        if self.use_fc:
            self.fc_out_dim = 512
            # Determine dimension:
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.fc_out_dim),
            )

        self.encode_projection = ContextMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                batch_norm=args.batch_norm,dropout=args.map_dropout,
                                pre_fc=args.map_pre_fc,ctx_head=args.ctx_head)
        
        inp_dim = args.map_dim
        self.adapt_relation = TaskAwareRelation(inp_dim=inp_dim, hidden_dim=args.rel_hidden_dim,
                                                num_layers=args.rel_layer, edge_n_layer=args.rel_edge_layer,
                                                top_k=args.rel_k, res_alpha=args.rel_res,
                                                batch_norm=args.batch_norm, adj_type=args.rel_adj,
                                                activation=args.rel_act, node_concat=args.rel_node_concat,dropout=args.rel_dropout,
                                                pre_dropout=args.rel_dropout2)

    def to_one_hot(self,class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)

    def label2edge(self, label, mask_diag=True):
        # get size
        num_samples = label.size(1)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # expand
        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2)).unsqueeze(0).unsqueeze(0).repeat(edge.size(0), 1, 1, 1).to(edge.device)
            edge=edge*diag_mask
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False,return_adj=False,return_emb=False):
        if not return_emb:
            s_logits, q_logits, adj = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        else:
            s_logits, q_logits, adj, s_rel_emb, q_rel_emb = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        if q_pred_adj:
            q_sim = adj[-1][:, 0, -1, :-1]
            q_logits = q_sim @ self.to_one_hot(s_label)
        if not return_emb:
            return s_logits, q_logits, adj
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb

    def forward(self, input_batch: DKTBatch, q_pred_adj=False):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            support_features.append(self.graph_feature_extractor(input_batch.support_features))
            query_features.append(self.graph_feature_extractor(input_batch.query_features))
        if "ecfp" in self.config.used_features:
            support_features.append(input_batch.support_features.fingerprints.float())
            query_features.append(input_batch.query_features.fingerprints.float())
        if "pc-descs" in self.config.used_features:
            support_features.append(input_batch.support_features.descriptors)
            query_features.append(input_batch.query_features.descriptors)

        support_features_flat = torch.cat(support_features, dim=1)
        query_features_flat = torch.cat(query_features, dim=1)

        if self.use_fc:
            support_features_flat = self.fc(support_features_flat)
            query_features_flat = self.fc(query_features_flat)

        if self.normalizing_features:
            support_features_flat = torch.nn.functional.normalize(support_features_flat, p=2, dim=1)
            query_features_flat = torch.nn.functional.normalize(query_features_flat, p=2, dim=1)

        if self.config.use_numeric_labels:
            raise NotImplementedError
        else:
            support_labels_converted = input_batch.support_labels
            query_labels_converted = input_batch.query_labels

        s_emb_map, q_emb_map = self.encode_projection(support_features_flat, query_features_flat)
        s_node_emb = None
        s_logits, q_logits, adj = self.relation_forward(s_emb_map, q_emb_map, support_labels_converted, q_pred_adj=q_pred_adj)

        return s_logits, q_logits, adj, s_node_emb
        
