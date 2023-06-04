import torch
import torch.nn as nn
import numpy as np

import gpytorch

from .encoder import GNN_Encoder

# Custom FS mol
import sys
sys.path.append("../")
from fs_mol.utils.gp_utils import ExactGPLayer


class ADKFModel(nn.Module):
    def __init__(self, args):
        super(ADKFModel, self).__init__()

        self.gp_kernel = "matern"
        self.emb_dim = args.emb_dim
        self.gpu_id = args.gpu_id

        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)

        self.__create_tail_GP(kernel_type=self.gp_kernel)

    
    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.startswith("gp_"):
            #if name.startswith("mol_encoder.gnn.gnns.4") or name.startswith("mol_encoder.gnn.batch_norms.4"):
                fe_params.append(param)
        return fe_params

    def gp_params(self):
        gp_params = []
        for name, param in self.named_parameters():
            if name.startswith("gp_"):
                gp_params.append(param)
        return gp_params


    def reinit_gp_params(self, gp_input, use_lengthscale_prior=True):

        self.__create_tail_GP(kernel_type=self.gp_kernel)

        if self.gp_kernel == 'matern' or self.gp_kernel == 'rbf' or self.gp_kernel == 'RBF':
            median_lengthscale_init = self.compute_median_lengthscale_init(gp_input)
            if use_lengthscale_prior:
                scale = 0.25
                loc = torch.log(median_lengthscale_init).item() + scale**2 # make sure that mode=median_lengthscale_init
                lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
                self.gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * median_lengthscale_init


    def __create_tail_GP(self, kernel_type):
        dummy_train_x = torch.ones(20, self.emb_dim)
        dummy_train_y = torch.ones(20)

        ard_num_dims = None

        scale = 0.25
        loc = np.log(0.1) + scale**2 # make sure that mode=0.1
        noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior).to(self.device)
        self.gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims, use_numeric_labels=False
        ).to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model).to(self.device)

        
    def compute_median_lengthscale_init(self, gp_input):
        dist_squared = torch.cdist(gp_input, gp_input) ** 2
        dist_squared = torch.triu(dist_squared, diagonal=1)
        return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


    def forward(self, s_data, q_data, train_loss: bool, s_label=None, q_pred_adj=False, predictive_val_loss: bool=False, is_functional_call: bool=False):
        support_features_flat, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        support_labels_converted = self.__convert_bool_labels(s_label)
        if q_data is not None:
            query_features_flat, _ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            query_labels_converted = self.__convert_bool_labels(q_data.y)

        # compute train/val loss if the model is in the training mode
        assert self.training
        assert train_loss is not None
        if train_loss: # compute train loss (on the support set)
            if is_functional_call: # return loss directly
                self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                logits = self.gp_model(support_features_flat)
                logits = -self.mll(logits, self.gp_model.train_targets)
            else:
                self.reinit_gp_params(support_features_flat.detach(), use_lengthscale_prior=True)
                self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_labels_converted.detach(), strict=False)
                logits = None
        else: # compute val loss (on the query set)
            assert is_functional_call == True
            if predictive_val_loss:
                self.gp_model.eval()
                self.gp_likelihood.eval()
                with gpytorch.settings.detach_test_caches(False):
                    self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                    # return sum of the log predictive losses for all data points, which converges better than averaged loss
                    logits = -self.gp_likelihood(self.gp_model(query_features_flat)).log_prob(query_labels_converted) #/ self.predictive_targets.shape[0]
                self.gp_model.train()
                self.gp_likelihood.train()
            else:
                # self.gp_model.set_train_data(inputs=query_features_flat, targets=query_labels_converted, strict=False)
                # logits = self.gp_model(query_features_flat)
                # logits = -self.mll(logits, self.gp_model.train_targets)
                raise NotImplementedError

        return logits


    def forward_query_loader(self, s_data, q_loader, train_loss: bool, s_label=None, q_pred_adj=False, predictive_val_loss: bool=False, is_functional_call: bool=False):
        
        # compute train/val loss if the model is in the training mode
        assert self.training == False
        
        support_features_flat, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        support_labels_converted = self.__convert_bool_labels(s_label)

        logits_list = []
        query_labels_converted_list = []

        # do GP posterior inference if the model is in the evaluation mode
        assert train_loss is None

        self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)

        for q_data in q_loader:
            q_data = q_data.to(support_features_flat.device)
            query_labels_converted_list.append(q_data.y) # NO NEED TO CONVERT to -1, 1
            query_features_flat, _ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            logits = self.gp_likelihood(self.gp_model(query_features_flat)).mean
            logits_list.append(logits)

        return torch.sigmoid(torch.cat(logits_list, 0)), torch.cat(query_labels_converted_list, 0)


    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0
