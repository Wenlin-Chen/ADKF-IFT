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

from fs_mol.utils.gp_utils import VariationalGPLayer, ExactGPLayer

import gpytorch
from gpytorch.distributions import MultivariateNormal

from botorch.optim.fit import fit_gpytorch_scipy

#from fs_mol.utils._stateless import functional_call

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class ADKTVGPModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    #distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"


class ADKTVGPModel(nn.Module):
    def __init__(self, config: ADKTVGPModelConfig):
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

        dummy_train_x = torch.ones(64, self.config.graph_feature_extractor_config.readout_config.output_dim)
        dummy_train_y = torch.ones(64)
        self.__create_tail_GP(train_x=dummy_train_x, train_y=dummy_train_y, kernel_type=self.config.gp_kernel)

        if self.config.gp_kernel == "cossim":
            self.normalizing_features = True
        else:
            self.normalizing_features = False

    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.startswith("gp_"):
                fe_params.append(param)
        return fe_params

    def gp_params(self):
        gp_params = []
        for name, param in self.named_parameters():
            if name.startswith("gp_model."):
                gp_params.append(param)
        return gp_params

    def reinit_gp_params(self, train_x, train_y, use_lengthscale_prior=False):

        train_y_regression = self.__convert_bool_labels_for_regression(train_y)
        exact_gp_model, exact_gp_likelihood, exact_mll = self.__create_exact_GP_for_init(kernel_type=self.config.gp_kernel)

        if self.config.gp_kernel == 'matern' or self.config.gp_kernel == 'rbf' or self.config.gp_kernel == 'RBF':
            median_lengthscale_init = self.compute_median_lengthscale_init(train_x)
            if use_lengthscale_prior:
                scale = 0.25
                loc = torch.log(median_lengthscale_init).item() + scale**2 # make sure that mode=median_lengthscale_init
                lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
                exact_gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )
            exact_gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(exact_gp_model.covar_module.base_kernel.lengthscale) * median_lengthscale_init
        
        exact_gp_model.set_train_data(inputs=train_x.detach(), targets=train_y_regression.detach(), strict=False)
        fit_gpytorch_scipy(exact_mll)

        self.__create_tail_GP(train_x=train_x, train_y=train_y, kernel_type=self.config.gp_kernel)

        if self.config.gp_kernel == 'matern' or self.config.gp_kernel == 'rbf' or self.config.gp_kernel == 'RBF':
            lengthscale_init = exact_gp_model.covar_module.base_kernel.lengthscale
            if use_lengthscale_prior:
                scale = 0.2
                loc = torch.log(lengthscale_init).item() + scale**2 # make sure that mode=lengthscale_init
                lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
                self.gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * lengthscale_init

        self.gp_model.covar_module.outputscale = exact_gp_model.covar_module.outputscale

    def __create_exact_GP_for_init(self, kernel_type):
        dummy_train_x = torch.ones(64, self.config.graph_feature_extractor_config.readout_config.output_dim)
        dummy_train_y = torch.ones(64)

        if self.config.use_ard:
            ard_num_dims = self.config.graph_feature_extractor_config.readout_config.output_dim
        else:
            ard_num_dims = None

        exact_gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        exact_gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=exact_gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims
        ).to(self.device)
        exact_mll = gpytorch.mlls.ExactMarginalLogLikelihood(exact_gp_likelihood, exact_gp_model).to(self.device)

        return exact_gp_model, exact_gp_likelihood, exact_mll

    def __create_tail_GP(self, train_x, train_y, kernel_type):
        if self.config.use_ard:
            ard_num_dims = self.config.graph_feature_extractor_config.readout_config.output_dim
        else:
            ard_num_dims = None

        self.gp_likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        self.gp_model = VariationalGPLayer(
            train_x=train_x, train_y=train_y, kernel=kernel_type, ard_num_dims=ard_num_dims
        ).to(self.device)
        self.mll = gpytorch.mlls.VariationalELBO(self.gp_likelihood, self.gp_model, train_y.numel()).to(self.device)

    def compute_median_lengthscale_init(self, gp_input):
        dist_squared = torch.cdist(gp_input, gp_input) ** 2
        dist_squared = torch.triu(dist_squared, diagonal=1)
        return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: DKTBatch, train_loss: bool, predictive_val_loss: bool=False, is_functional_call: bool=False):
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
                if is_functional_call: # return loss directly
                    #self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                    logits = self.gp_model(support_features_flat)
                    logits = -self.mll(logits, self.support_labels_converted)
                else:
                    self.reinit_gp_params(support_features_flat.detach(), support_labels_converted.detach(), self.config.use_lengthscale_prior)
                    self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_labels_converted.detach(), strict=False)
                    with torch.no_grad():
                        logits = self.gp_model(support_features_flat.detach()) # I don't know why, but this is a necessary step before calling fit_gpytorch_scipy when using VariationalELBO
            else: # compute val loss (on the query set)
                assert is_functional_call == True
                assert predictive_val_loss == True
                self.gp_model.eval()
                self.gp_likelihood.eval()
                with gpytorch.settings.detach_test_caches(False):
                    #self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                    self.gp_model.variational_strategy.register_buffer("inducing_points", support_features_flat) # backprop through the support points
                    # return sum of the log predictive losses for all data points, which converges better than averaged loss
                    logits = -self.gp_likelihood(self.gp_model(query_features_flat)).log_prob(query_labels_converted).sum() #/ query_features_flat.shape[0]
                self.gp_model.train()
                self.gp_likelihood.train()

        # do GP posterior inference if the model is in the evaluation mode
        else:
            assert train_loss is None
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)

            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))

        return logits

    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> 0.0
        return labels.float()

    def __convert_bool_labels_for_regression(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0
