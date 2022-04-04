import sys
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

import pandas as pd
from fs_mol.data.fsmol_task import MoleculeDatapoint, GraphData
from fs_mol.utils.gp_utils import TanimotoKernel
from fs_mol.preprocessing.featurisers.molgraph_utils import molecule_to_graph

from fs_mol.data import (
    FSMolTask,
    FSMolBatch,
    FSMolBatcher,
    MoleculeDatapoint,
)
from fs_mol.data.dkt import (
    FeaturisedDKTTaskSample,
    DKTBatch,
    MoleculeDKTFeatures,
)
from fs_mol.utils.adaptive_dkt_utils import ADKTModelTrainer, ADKTModelTrainerConfig
from fs_mol.utils.dkt_utils import DKTModelTrainer, DKTModelTrainerConfig

from rdkit import DataStructs
from rdkit.Chem import (
    Mol,
    MolFromSmiles,
    rdFingerprintGenerator,
    CanonSmiles,
    Descriptors,
)

import matplotlib.pyplot as plt
import numpy as np
import math

import torch
import gpytorch
import botorch
from botorch.optim.fit import fit_gpytorch_scipy

from dpu_utils.utils import RichPath


def get_feature_extractors_from_metadata(metadata_path, metadata_filename="metadata.pkl.gz"):
    metapath = RichPath.create(metadata_path)
    path = metapath.join(metadata_filename)
    metadata = path.read_by_file_suffix()
    return metadata["feature_extractors"]


def unit_factor(unit):
    """Return the factor corresponding to the unit, e.g. 1E-9 for nM.
    Known units are: mM, uM, nM, pM. Raises ValueError for unknown unit."""
    units = ["mm", "um", "nm", "pm"]
    pos = units.index(unit.lower()) + 1
    factor = 10 ** -(pos * 3)
    return factor


def pic50(ic50, unit="um"):
    """Calculate pIC50 from IC50. Optionally, a unit for the input IC50 value may be given.
    Known units are: mM, uM, nM, pM"""
    if unit is not None:
        ic50 *= unit_factor(unit)
    return float(-math.log10(ic50))


def load_antibiotics_dataset(xlsx_file, metadata_path):
    df = pd.read_excel(xlsx_file, sheet_name="S1B", header=1)
    dataset = []

    # get pre-defined atom_feature_extractors from metadata provided in FS-Mol
    atom_feature_extractors = get_feature_extractors_from_metadata(metadata_path)

    for i, row in df.iterrows():

        # get molecule info from the xlsx file
        numeric_label = float(row["Mean_Inhibition"])
        smiles = CanonSmiles(row["SMILES"].strip())
        activity = row["Activity"]
        bool_label = True if activity=="Active" else False
        
        # get fingerprint
        rdkit_mol = MolFromSmiles(smiles)
        fp_vec = rdFingerprintGenerator.GetCountFPs([rdkit_mol], fpType=rdFingerprintGenerator.MorganFP)[0]
        fp_numpy = np.zeros((0,), np.int8) 
        DataStructs.ConvertToNumpyArray(fp_vec, fp_numpy)
        
        # get graph
        try:
            graph_dict = molecule_to_graph(rdkit_mol, atom_feature_extractors)
            adjacency_lists = []
            for adj_list in graph_dict["adjacency_lists"]:
                if adj_list:
                    adjacency_lists.append(np.array(adj_list, dtype=np.int64))
                else:
                    adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))
            graph = GraphData(
                node_features=np.array(graph_dict["node_features"], dtype=np.float32), 
                adjacency_lists=adjacency_lists, 
                edge_features=[]
            )
        except IndexError:
            print(
                f"Skipping datapoint {smiles}, cannot featurise with current metadata."
            )
            continue

        # get descriptors
        descriptors = []
        for descr in Descriptors._descList:
            _, descr_calc_fn = descr
            descriptors.append(descr_calc_fn(rdkit_mol))
        descriptors = np.array(descriptors, dtype=np.float32)
        
        # create a MoleculeDatapoint object
        mol = MoleculeDatapoint(
            task_name="antibiotics", smiles=smiles, graph=graph, 
            numeric_label=numeric_label, bool_label=bool_label, 
            fingerprint=fp_numpy, descriptors=descriptors)
        dataset.append(mol)

    return FSMolTask(name="antibiotics", samples=dataset)


def load_covid_moonshot_dataset(csv_file, metadata_path):
    df = pd.read_csv(csv_file, sep=",", header=0)
    df = df.sort_values(by=["f_avg_IC50"], ascending=True)
    dataset = []

    # get pre-defined atom_feature_extractors from metadata provided in FS-Mol
    atom_feature_extractors = get_feature_extractors_from_metadata(metadata_path)

    for i, row in df.iterrows():

        # get molecule info from the csv file
        raw_numeric_label = float(row["f_avg_IC50"])
        smiles = CanonSmiles(row["SMILES"].strip())
        if np.isnan(raw_numeric_label):
            print(
                f"Skipping datapoint {smiles} (IC50 Fluorescence not available)."
            )
            continue

        numeric_label = float(-1.0 * pic50(raw_numeric_label))

        bool_label = True if raw_numeric_label < 5.0 else False
        
        # get fingerprint
        rdkit_mol = MolFromSmiles(smiles)
        fp_vec = rdFingerprintGenerator.GetCountFPs([rdkit_mol], fpType=rdFingerprintGenerator.MorganFP)[0]
        fp_numpy = np.zeros((0,), np.int8) 
        DataStructs.ConvertToNumpyArray(fp_vec, fp_numpy)
        
        # get graph
        try:
            graph_dict = molecule_to_graph(rdkit_mol, atom_feature_extractors)
            adjacency_lists = []
            for adj_list in graph_dict["adjacency_lists"]:
                if adj_list:
                    adjacency_lists.append(np.array(adj_list, dtype=np.int64))
                else:
                    adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))
            graph = GraphData(
                node_features=np.array(graph_dict["node_features"], dtype=np.float32), 
                adjacency_lists=adjacency_lists, 
                edge_features=[]
            )
        except IndexError:
            print(
                f"Skipping datapoint {smiles}, cannot featurise with current metadata."
            )
            continue

        # get descriptors
        descriptors = []
        for descr in Descriptors._descList:
            _, descr_calc_fn = descr
            descriptors.append(descr_calc_fn(rdkit_mol))
        descriptors = np.array(descriptors, dtype=np.float32)
        
        # create a MoleculeDatapoint object
        mol = MoleculeDatapoint(
            task_name="covid_moonshot", smiles=smiles, graph=graph, 
            numeric_label=numeric_label, bool_label=bool_label, 
            fingerprint=fp_numpy, descriptors=descriptors)
        dataset.append(mol)

    return FSMolTask(name="covid_moonshot", samples=dataset)


def task_to_batches(
    task: FSMolTask, batcher: FSMolBatcher[MoleculeDKTFeatures, np.ndarray]
):

    batches = []
    for features, labels, numeric_labels in batcher.batch(task.samples):
        batches.append(features)

    return batches


# Minimizing; data points should be sorted according to the value of y_all in the ascending order
def run_gp_ei_bo(dataset, x_all, y_all, num_init_points, query_batch_size, num_bo_iters, kernel_type, device, init_from, noise_init=0.01, noise_prior=True):
    
    y_mean = y_all.mean()
    y_std = y_all.std()
    y_all = (y_all - y_mean) / y_std
    
    bo_record = []
    queried_idx = []
    
    init_points_idx = np.random.choice(np.arange(init_from, len(dataset)), size=num_init_points, replace=False).tolist()
    queried_idx.extend(init_points_idx)
    
    x_queried, y_queried = x_all[queried_idx, :], y_all[queried_idx]
    best_y_queried = y_queried.min().item()
    bo_record.append(min(queried_idx))
    
    for i in range(num_bo_iters):
        likelihood, model, mll = create_gp(x_queried, y_queried, kernel_type, device, noise_init, noise_prior)
        model.train()
        likelihood.train()
        fit_gpytorch_scipy(mll)
        model.eval()
        likelihood.eval()
        acq_func = botorch.acquisition.analytic.ExpectedImprovement(model, best_y_queried, maximize=False)
        
        acq_values = []
        for j in range(len(dataset)):
            if j in queried_idx:
                acq_value = -np.inf
            else:
                acq_value = acq_func(x_all[j:j+1, :]).item()
            acq_values.append(acq_value)
        acq_values = torch.tensor(acq_values)
        
        num_nonzero_acq = torch.sum(acq_values>0)
        if num_nonzero_acq == 0:
            query_idx = np.random.choice([i for i in range(len(dataset)) if i not in queried_idx], size=query_batch_size, replace=False)
            query_idx = query_idx.tolist()
        elif num_nonzero_acq > 0 and num_nonzero_acq < query_batch_size:
            _, query_idx = torch.topk(acq_values, query_batch_size)
            query_idx = query_idx[:num_nonzero_acq].tolist()
            query_idx2 = np.random.choice([i for i in range(len(dataset)) if i not in queried_idx+query_idx], size=query_batch_size-num_nonzero_acq, replace=False)
            query_idx.extend(query_idx2.tolist())
        else:
            _, query_idx = torch.topk(acq_values, query_batch_size)
            query_idx = query_idx.tolist()
        queried_idx.extend(query_idx)
        queried_idx = list(set(queried_idx))
        
        x_queried, y_queried = x_all[queried_idx, :], y_all[queried_idx]
        best_y_queried = y_queried.min().item()
        
        for j in query_idx[::-1]:
            bo_record.append(j)
            
    return bo_record


class CustomKernelGP(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1  # botorch needs this

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        kernel: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.Likelihood,
    ):

        botorch.models.gpytorch.GPyTorchModel.__init__(self)
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)

        self.covar_module = kernel
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def create_gp(train_x, train_y, kernel_type, device, noise_init=0.01, noise_prior=True):

    if noise_prior:
        scale = 0.25
        loc = np.log(noise_init) + scale**2 # make sure that mode=noise
        noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
    else:
        noise_prior = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior).to(device)
    likelihood.noise_covar.noise = noise_init

    if kernel_type == "tanimoto":
        kernel = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    elif kernel_type == "matern":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

        median_lengthscale_init = compute_median_lengthscale_init(train_x)
        scale = 0.25
        loc = torch.log(median_lengthscale_init).item() + scale**2 # make sure that mode=median_lengthscale_init
        lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        kernel.base_kernel.register_prior(
            "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
        )
        kernel.base_kernel.lengthscale = torch.ones_like(kernel.base_kernel.lengthscale).to(device) * median_lengthscale_init
    
    else:
        raise ValueError
    model = CustomKernelGP(train_x, train_y, kernel, likelihood).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)
    
    return likelihood, model, mll


def compute_median_lengthscale_init(gp_input):
    dist_squared = torch.cdist(gp_input, gp_input) ** 2
    dist_squared = torch.triu(dist_squared, diagonal=1)
    return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))


def min_so_far(x):
    y = [x[0]]
    curr_min = x[0]
    for i in range(1, len(x)):
        if x[i] < curr_min:
            curr_min = x[i]
        y.append(curr_min)

    return y
    

class ADKTModelFeatureExtractor(ADKTModelTrainer):
    def __init__(self, config: ADKTModelTrainerConfig):
        super().__init__(config)

    def get_representation(self, features):

        with torch.no_grad():
            representation = []

            if "gnn" in self.config.used_features:
                representation.append(self.graph_feature_extractor(features))
            if "ecfp" in self.config.used_features:
                representation.append(features.fingerprints)
            if "pc-descs" in self.config.used_features:
                representation.append(features.descriptors)

            representation = torch.cat(representation, dim=1)

            if self.use_fc:
                representation= self.fc(representation)

            if self.normalizing_features:
                representation = torch.nn.functional.normalize(representation, p=2, dim=1)

        return representation

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        quiet: bool = True,
        device: torch.device = None,
    ):
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        model = ADKTModelFeatureExtractor(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            #load_task_specific_weights=True,
            device=device,
        )
        return model


class DKTModelFeatureExtractor(DKTModelTrainer):
    def __init__(self, config: DKTModelTrainerConfig):
        super().__init__(config)

    def get_representation(self, features):

        with torch.no_grad():
            representation = []

            if "gnn" in self.config.used_features:
                representation.append(self.graph_feature_extractor(features))
            if "ecfp" in self.config.used_features:
                representation.append(features.fingerprints)
            if "pc-descs" in self.config.used_features:
                representation.append(features.descriptors)

            representation = torch.cat(representation, dim=1)

            if self.use_fc:
                representation= self.fc(representation)

            if self.normalizing_features:
                representation = torch.nn.functional.normalize(representation, p=2, dim=1)

        return representation

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        quiet: bool = True,
        device: torch.device = None,
    ):
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        model = DKTModelFeatureExtractor(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            #load_task_specific_weights=True,
            device=device,
        )
        return model
