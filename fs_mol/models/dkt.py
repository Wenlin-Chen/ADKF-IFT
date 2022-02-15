from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
from fs_mol.data.dkt import DKTBatch

from fs_mol.utils.gp_utils import ExactGPLayer

import gpytorch
from gpytorch.distributions import MultivariateNormal

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class DKTModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    #distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class DKTModel(nn.Module):
    def __init__(self, config: DKTModelConfig):
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

        kernel_type = self.config.gp_kernel
        if self.config.use_ard:
            ard_num_dims = self.config.graph_feature_extractor_config.readout_config.output_dim
        else:
            ard_num_dims = None
        self.__create_tail_GP(kernel_type=kernel_type, ard_num_dims=ard_num_dims, use_lengthscale_prior=self.config.use_lengthscale_prior)

        if kernel_type == "cossim":
            self.normalizing_features = True
        else:
            self.normalizing_features = False

    def __create_tail_GP(self, kernel_type, ard_num_dims=None, use_lengthscale_prior=False):
        dummy_train_x = torch.ones(64, self.config.graph_feature_extractor_config.readout_config.output_dim)
        dummy_train_y = torch.ones(64)

        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims, use_lengthscale_prior=use_lengthscale_prior
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: DKTBatch):
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

        if self.normalizing_features:
            support_features_flat = torch.nn.functional.normalize(support_features_flat, p=2, dim=1)
            query_features_flat = torch.nn.functional.normalize(query_features_flat, p=2, dim=1)

        support_labels_converted = self.__convert_bool_labels(input_batch.support_labels)
        query_labels_converted = self.__convert_bool_labels(input_batch.query_labels)

        if self.training:
            combined_features_flat = torch.cat([support_features_flat, query_features_flat], dim=0)
            combined_labels_converted = torch.cat([support_labels_converted, query_labels_converted])

            self.gp_model.set_train_data(inputs=combined_features_flat, targets=combined_labels_converted, strict=False)
            logits = self.gp_model(combined_features_flat)
        else:
            self.gp_model.train()
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
            self.gp_model.eval()

            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))

        return logits

    def compute_loss(self, logits: MultivariateNormal) -> torch.Tensor:
        assert self.training == True
        return -self.mll(logits, self.gp_model.train_targets)

    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0
