from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
from fs_mol.data.cnp import CNPBatch

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class CNPModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    #distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class CNPModel(nn.Module):
    def __init__(self, config: CNPModelConfig):
        super().__init__()
        self.config = config

        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                config.graph_feature_extractor_config
            )

        self.use_fc = self.config.used_features.endswith("+fc")

        # Create MLP if needed:
        if self.use_fc:
            # Determine dimension:
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.config.graph_feature_extractor_config.readout_config.output_dim),
            )

        self.encoder_label_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.encoder_final_fc = nn.Sequential(
            nn.Linear(64+self.config.graph_feature_extractor_config.readout_config.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(128+self.config.graph_feature_extractor_config.readout_config.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: CNPBatch):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            support_features.append(self.graph_feature_extractor(input_batch.support_features))
            query_features.append(self.graph_feature_extractor(input_batch.query_features))
        if "ecfp" in self.config.used_features:
            support_features.append(input_batch.support_features.fingerprints)
            query_features.append(input_batch.query_features.fingerprints)
        if "pc-descs" in self.config.used_features:
            support_features.append(input_batch.support_features.descriptors)
            query_features.append(input_batch.query_features.descriptors)

        support_features_flat = torch.cat(support_features, dim=1)
        query_features_flat = torch.cat(query_features, dim=1)

        if self.use_fc:
            support_features_flat = self.fc(support_features_flat)
            query_features_flat = self.fc(query_features_flat)

        if self.config.use_numeric_labels:
            support_labels_converted = input_batch.support_numeric_labels.float().unsqueeze(1)
            query_labels_converted = input_batch.query_numeric_labels.float().unsqueeze(1)
        else:
            support_labels_converted = self.__convert_bool_labels(input_batch.support_labels).unsqueeze(1)
            query_labels_converted = self.__convert_bool_labels(input_batch.query_labels).unsqueeze(1)

        support_labels_embedding = self.encoder_label_fc(support_labels_converted)
        support_pairs = torch.cat([support_features_flat, support_labels_embedding], dim=1)
        support_pairs_embedding = self.encoder_final_fc(support_pairs)
        representation = support_pairs_embedding.mean(dim=0, keepdim=True).repeat([query_features_flat.shape[0], 1])

        query_representation_pairs = torch.cat([representation, query_features_flat], dim=1)
        decoder_output = self.decoder_fc(query_representation_pairs)
        mu, log_sigma = torch.split(decoder_output, 1, dim=1)
        sigma = 0.01 + 0.09 * torch.nn.functional.softplus(log_sigma)

        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        log_prob = dist.log_prob(query_labels_converted)

        return log_prob, mu, sigma

    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0
