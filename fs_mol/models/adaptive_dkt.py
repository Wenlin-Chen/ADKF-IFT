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

from copy import deepcopy

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class ADKTModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    #distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class ADKTModel(nn.Module):
    def __init__(self, config: ADKTModelConfig):
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
        self.save_gp_states()

        if kernel_type == "cossim":
            self.normalizing_features = True
        else:
            self.normalizing_features = False

    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.startswith("gp_"):
                fe_params.append(param)
        return fe_params

    def reinit_gp_params(self):
        self.gp_model.load_state_dict(self.cached_gp_model_state)
        self.gp_likelihood.load_state_dict(self.cached_gp_likelihood_state)

    def save_gp_states(self):
        self.cached_gp_model_state = deepcopy(self.gp_model.state_dict())
        self.cached_gp_likelihood_state = deepcopy(self.gp_likelihood.state_dict())

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

    def forward(self, input_batch: DKTBatch, train_loss: bool, predictive_val_loss: bool=False):
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

        # compute train/val loss if the model is in the training mode
        if self.training:
            assert train_loss is not None
            if train_loss: # compute train loss (on the support set)
                self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_labels_converted.detach(), strict=False)
                logits = None
            else: # compute val loss (on the query set)
                if predictive_val_loss:
                    self.gp_model.eval()
                    with gpytorch.settings.detach_test_caches(False):
                        self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                        logits = self.gp_model(query_features_flat)
                    self.predictive_targets = query_labels_converted
                else:
                    self.gp_model.set_train_data(inputs=query_features_flat, targets=query_labels_converted, strict=False)
                    logits = self.gp_model(query_features_flat)

        # do GP posterior inference if the model is in the evaluation mode
        else:
            assert train_loss is None
            self.gp_model.train()
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
            self.gp_model.eval()

            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))

        return logits

    def compute_loss(self, logits: MultivariateNormal, predictive: bool=False) -> torch.Tensor:
        assert self.training == True
        if predictive:
            with gpytorch.settings.detach_test_caches(False):
                predictive_loss = -self.gp_likelihood(logits).log_prob(self.predictive_targets)
            self.gp_model.train()
            return predictive_loss
        else:
            return -self.mll(logits, self.gp_model.train_targets)

    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0
