import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))


from fs_mol.models.abstract_torch_fsmol_model import linear_warmup
from fs_mol.data import FSMolDataset, FSMolTaskSample, DataFold
from fs_mol.data.dkt import (
    DKTBatch,
    get_dkt_task_sample_iterable,
    get_dkt_batcher,
    task_sample_to_dkt_task_sample,
)
from fs_mol.models.par import PARModel, PARModelConfig
from fs_mol.models.abstract_torch_fsmol_model import MetricType
from fs_mol.utils.metrics import (
    compute_binary_task_metrics,
    avg_metrics_over_tasks,
    avg_task_metrics_list,
    compute_numeric_task_metrics,
    avg_numeric_metrics_over_tasks,
    avg_task_numeric_metrics_list
)
from fs_mol.utils.metric_logger import MetricLogger
from fs_mol.utils.torch_utils import torchify
from fs_mol.utils.test_utils import eval_model

from botorch.optim.fit import fit_gpytorch_scipy

from fs_mol.utils.cauchy_hypergradient import cauchy_hypergradient
from fs_mol.utils.cauchy_hypergradient_jvp import cauchy_hypergradient_jvp
from fs_mol.utils._stateless import functional_call

# Chem lib stuff
from chem_lib.models.maml import MAML


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PARModelTrainerConfig(PARModelConfig):
    batch_size: int = 256
    tasks_per_batch: int = 9  # from their code
    support_set_size: int = 16
    query_set_size: int = 256

    num_train_steps: int = 10000
    validate_every_num_steps: int = 50
    validation_support_set_sizes: Tuple[int] = (16, 128)
    validation_query_set_size: int = 256
    validation_num_samples: int = 5

    # Optimization params
    outer_learning_rate: float = 0.001
    inner_learning_rate: float = 0.001
    clip_value: Optional[float] = None
    weight_decay: float = 5e-5
    num_updates_per_batch: int = 1
    num_inner_update_step: int = 1
    reg_adj: float = 1.0  # From their code

    # Architecture
    emb_dim: int = 300
    map_dim: int = 128
    map_layer: int = 2
    batch_norm: bool = False
    map_dropout: float = 0.1
    map_pre_fc: int = 0
    ctx_head: int = 2
    rel_dropout2: float = 0.2
    rel_dropout: float = 0.0
    rel_node_concat: bool = False
    rel_act: str = "sigmoid"
    rel_adj: str = "sim"
    rel_res: float = 0.0
    rel_k: int = -1
    rel_hidden_dim: int = 128
    rel_layer: int = 2
    rel_edge_layer: int = 2

    # MAML
    second_order_maml: bool = True

    use_numeric_labels: bool = False

def get_predictions(model, data_batch: DKTBatch, train=True, **kwargs):
    """Make PAR predictions"""

    # if train:
    s_logits, q_logits, adj, s_node_emb = model(input_batch=data_batch, **kwargs)
    pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj,}

    return pred_dict

def get_loss(model: MAML, pred_dict: dict, train: bool, flag: bool, batch_features: DKTBatch):
    """Loss for the PAR model"""
    criterion = torch.nn.CrossEntropyLoss().to(pred_dict["s_logits"])

    # Cast appropriate things to int
    support_labels = batch_features.support_labels.to(torch.long)
    query_labels = batch_features.query_labels.to(torch.long)

    # Define their n_query/etc variables
    n_query = len(batch_features.query_labels)

    # Simplified version of their logic
    if train and not flag:
        # losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])
        losses_adapt = criterion(
            input=pred_dict['q_logits'], 
            target=query_labels,
        )
    else:
        losses_adapt = criterion(
            input=pred_dict['s_logits'].reshape(-1,2), 
            target=support_labels.repeat(n_query)
        )

    # This whole block we did not modify, except for changing "batch_data" variables
    if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
        print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
        print(pred_dict['s_logits'])
        losses_adapt = torch.zeros_like(losses_adapt)
    if model.config.reg_adj > 0:
        n_support = len(batch_features.support_labels)
        adj = pred_dict['adj'][-1]
        if train:
            if flag:
                s_label = support_labels.unsqueeze(0).repeat(n_query, 1)
                n_d = n_query * n_support
                label_edge = model.label2edge(s_label).reshape((n_d, -1))
                pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
            else:
                s_label = support_labels.unsqueeze(0).repeat(n_query, 1)
                q_label = query_labels.unsqueeze(1)
                total_label = torch.cat((s_label, q_label), 1)
                label_edge = model.label2edge(total_label)[:,:,-1,:-1]
                pred_edge = adj[:,:,-1,:-1]
        else:
            s_label = support_labels.unsqueeze(0)
            n_d = n_support
            label_edge = model.label2edge(s_label).reshape((n_d, -1))
            pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
        adj_loss_val = F.mse_loss(pred_edge, label_edge)
        if torch.isnan(adj_loss_val).any() or torch.isinf(adj_loss_val).any():
            print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
            adj_loss_val = torch.zeros_like(adj_loss_val)

        losses_adapt += model.config.reg_adj * adj_loss_val

    return losses_adapt
    
def get_adaptable_weights(model):
    """Hard coded version of the '5' setting in their code."""

    # Note: no warmup
    fenc = lambda x: x[0]== 'graph_feature_extractor' or x[0] == "enc_fc"
    fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
    fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
    flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
    adaptable_weights = []
    adaptable_names=[]
    for name, p in model.module.named_parameters():
        names=name.split('.')
        if flag(names):
            adaptable_weights.append(p)
            adaptable_names.append(name)
    return adaptable_weights

def run_on_batches(
    maml_model: MAML,
    batches: List[DKTBatch],
    batch_labels: List[torch.Tensor],
    batch_numeric_labels: List[torch.Tensor],
    train: bool = False,
    #tasks_per_batch: int = 1,
):
    """Does inner loop adaptation."""

    if train:
        assert len(batches) == 1

    total_loss, total_num_samples = 0.0, 0
    task_preds: List[np.ndarray] = []
    task_labels: List[np.ndarray] = []

    #num_gradient_accumulation_steps = len(batches) * tasks_per_batch
    cloned_models = []
    for batch_features, this_batch_labels, this_batch_numeric_labels in zip(batches, batch_labels, batch_numeric_labels):
        
        model = maml_model.clone()
        model.train()
        adaptable_weights = get_adaptable_weights(model)
        cloned_models.append(model)
                        
        # MAML adaptation
        for inner_step in range(model.config.num_inner_update_step):
            pred_adapt = get_predictions(model=model, data_batch=batch_features, train=True)
            loss_adapt = get_loss(model=model, pred_dict=pred_adapt, train=True, flag = True, batch_features=batch_features)
            model.adapt(loss_adapt, adaptable_weights = adaptable_weights)
        
        # Compute loss on the support set at test time
        if not train:
            model.eval()
            with torch.no_grad():
                pred_eval = get_predictions(model=model, data_batch=batch_features, train=False)

                if model.config.use_numeric_labels:
                    raise NotImplementedError
                else:
                    batch_preds = F.softmax(pred_eval['logits'],dim=-1).detach()[:,1]
                    task_labels.append(this_batch_labels.detach().cpu().numpy())
                task_preds.append(batch_preds)
        

    if train:
        metrics = None
    else:
        predictions = np.concatenate(task_preds, axis=0)
        labels=np.concatenate(task_labels, axis=0)
        if model.config.use_numeric_labels:
            metrics = compute_numeric_task_metrics(predictions=predictions, labels=labels)
        else:
            metrics = compute_binary_task_metrics(predictions=predictions, labels=labels)

    return cloned_models, metrics


def evaluate_par_model(
    model: PARModel,
    dataset: FSMolDataset,
    support_sizes: List[int] = [16, 128],
    num_samples: int = 5,
    seed: int = 0,
    batch_size: int = 320,
    query_size: Optional[int] = None,
    data_fold: DataFold = DataFold.TEST,
    save_dir: Optional[str] = None,
):

    batcher = get_dkt_batcher(max_num_graphs=batch_size)

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ):
        dkt_task_sample = torchify(
            task_sample_to_dkt_task_sample(task_sample, batcher, model.config.use_numeric_labels), device=model.device
        )

        result_metrics = run_on_batches(
            model,
            batches=dkt_task_sample.batches,
            batch_labels=dkt_task_sample.batch_labels,
            batch_numeric_labels=dkt_task_sample.batch_numeric_labels,
            train=False,
        )
        
        if model.config.use_numeric_labels:
            logger.info(
                f"{dkt_task_sample.task_name}:"
                f" {dkt_task_sample.num_support_samples:3d} support samples,"
                f" {dkt_task_sample.num_query_samples:3d} query samples."
                f" R2 {result_metrics.r2:.5f}.",
            )
        else:
            logger.info(
                f"{dkt_task_sample.task_name}:"
                f" {dkt_task_sample.num_support_samples:3d} support samples,"
                f" {dkt_task_sample.num_query_samples:3d} query samples."
                f" Avg. prec. {result_metrics.avg_precision:.5f}.",
            )

        return result_metrics

    return eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=support_sizes,
        out_dir=save_dir,
        num_samples=num_samples,
        test_size_or_ratio=query_size,
        fold=data_fold,
        seed=seed,
        filter_numeric_labels=model.config.use_numeric_labels,
    )


def validate_by_finetuning_on_tasks(
    model: PARModel,
    dataset: FSMolDataset,
    seed: int = 0,
    aml_run=None,
    metric_to_use: MetricType = "avg_precision",
) -> float:
    """
    Validation function for PARModel. Similar to test function;
    each validation task is used to evaluate the model more than once, the
    final results are a mean value for all tasks over the requested metric.
    """

    task_results = evaluate_par_model(
        model,
        dataset,
        support_sizes=model.config.validation_support_set_sizes,
        num_samples=model.config.validation_num_samples,
        seed=seed,
        batch_size=model.config.batch_size,
        query_size=model.config.validation_query_set_size,
        data_fold=DataFold.VALIDATION,
    )

    # take the dictionary of task_results and return correct mean over all tasks
    if model.config.use_numeric_labels:
        mean_metrics = avg_numeric_metrics_over_tasks(task_results)
    else:
        mean_metrics = avg_metrics_over_tasks(task_results)
    if aml_run is not None:
        for metric_name, (metric_mean, _) in mean_metrics.items():
            aml_run.log(f"valid_task_test_{metric_name}", float(metric_mean))

    return mean_metrics[metric_to_use][0]


class PARModelTrainer(PARModel):
    def __init__(self, config: PARModelTrainerConfig):
        super().__init__(config)
        self.config = config
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def save_model(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
    ):
        data = self.get_model_state()

        if optimizer is not None:
            data["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            data["epoch"] = epoch

        torch.save(data, path)

    def load_model_weights(
        self,
        path: str,
        #load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        for name, param in pretrained_state_dict["model_state_dict"].items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            self.state_dict()[name].copy_(param)

        optimizer_weights = pretrained_state_dict.get("optimizer_state_dict")
        if optimizer_weights is not None:
            for name, param in optimizer_weights.items():
                self.optimizer.state_dict()[name].copy_(param)

    def load_model_gnn_weights(
        self,
        path: str,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        gnn_model_state_dict = pretrained_state_dict["model_state_dict"]
        our_state_dict = self.state_dict()

        # Load parameters (names specialised to GNNMultitask model), but also collect
        # parameters for GNN parts / rest, so that we can create a LR warmup schedule:
        gnn_params, other_params = [], []
        gnn_feature_extractor_param_name = "graph_feature_extractor."
        for our_name, our_param in our_state_dict.items():
            if (
                our_name.startswith(gnn_feature_extractor_param_name)
                and "final_norm_layer" not in our_name
            ):
                generic_name = our_name[len(gnn_feature_extractor_param_name) :]
                if generic_name.startswith("readout_layer."):
                    generic_name = f"readout{generic_name[len('readout_layer'):]}"
                our_param.copy_(gnn_model_state_dict[generic_name])
                logger.debug(f"I: Loaded parameter {our_name} from {generic_name} in {path}.")
                gnn_params.append(our_param)
            else:
                logger.debug(f"I: Not loading parameter {our_name}.")
                other_params.append(our_param)

        self.optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": self.config.learning_rate},
                {"params": gnn_params, "lr": self.config.learning_rate / 10},
            ],
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=[
                partial(linear_warmup, warmup_steps=0),  # for all params
                partial(linear_warmup, warmup_steps=100),  # for loaded GNN params
            ],
        )

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> "PARModelTrainer":
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        if not quiet:
            logger.info(f" Loading model configuration from {model_file}.")

        model = PARModelTrainer(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            #load_task_specific_weights=True,
            device=device,
        )
        return model

    def train_loop(self, out_dir: str, dataset: FSMolDataset, device: torch.device, aml_run=None):
        self.save_model(os.path.join(out_dir, "best_validation.pt"))

        train_task_sample_iterator = iter(
            get_dkt_task_sample_iterable(
                dataset=dataset,
                data_fold=DataFold.TRAIN,
                num_samples=1,
                max_num_graphs=self.config.batch_size,
                support_size=self.config.support_set_size,
                query_size=self.config.query_set_size,
                repeat=True,
                filter_numeric_labels=self.config.use_numeric_labels,
            )
        )

        best_validation_score = -np.inf
        metric_logger = MetricLogger(
            log_fn=lambda msg: logger.info(msg),
            aml_run=aml_run,
            window_size=max(10, self.config.validate_every_num_steps / 5),
        )
        
        # Define model and optimizer
        maml_model = MAML(self, lr=self.config.inner_learning_rate, first_order=not self.config.second_order_maml, anil=False, allow_unused=True)
        
        # Optimizer copied from their code
        self.optimizer = torch.optim.AdamW(maml_model.parameters(), lr=self.config.outer_learning_rate, weight_decay=self.config.weight_decay)

        # Overall outer loop steps
        for step in range(1, self.config.num_train_steps + 1):
            
            # Number of repeated steps on this batch
            for update_on_current_batch_idx in range(self.config.num_updates_per_batch):

                # Do inner loop on batch, one task at a time
                pred_losses = []
                for _ in range(self.config.tasks_per_batch):
                    task_sample = next(train_task_sample_iterator)
                    train_task_sample = torchify(task_sample, device=device)
                    batches = train_task_sample.batches
                    batch_labels = train_task_sample.batch_labels
                    batch_numeric_labels = train_task_sample.batch_numeric_labels
                    cloned_models, _ = run_on_batches(
                        maml_model,
                        batches=batches,
                        batch_labels=batch_labels,
                        batch_numeric_labels=batch_numeric_labels,
                        train=True,
                        #tasks_per_batch=self.config.tasks_per_batch,
                    )

                    # Make model predictions on query set
                    assert len(batches) == len(cloned_models) == 1
                    pred_eval = get_predictions(
                        model=cloned_models[0], data_batch=batches[0], train=True
                        )
                    pred_loss = get_loss(
                        model=cloned_models[0], pred_dict=pred_eval, train=True,
                        flag=False, batch_features=batches[0]
                    )
                    pred_losses.append(pred_loss)
                    del pred_loss
                
                # Outer loop update
                pred_losses_tensor = torch.stack(pred_losses)
                overall_loss = torch.sum(pred_losses_tensor) / len(pred_losses)
                self.optimizer.zero_grad()
                overall_loss.backward()
                torch.nn.utils.clip_grad_norm_(maml_model.parameters(), 1)
                self.optimizer.step()

                # print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())

            task_batch_mean_loss = overall_loss.detach().cpu().item() #np.mean(pred_losses_tensor.detach().cpu().numpy())
            #task_batch_avg_metrics = avg_task_metrics_list(task_batch_metrics)
            metric_logger.log_metrics(
                loss=task_batch_mean_loss,
                #avg_prec=task_batch_avg_metrics["avg_precision"][0],
                #kappa=task_batch_avg_metrics["kappa"][0],
                #acc=task_batch_avg_metrics["acc"][0],
            )

            if self.config.use_numeric_labels:
                metric_to_use = "r2"
            else:
                metric_to_use = "avg_precision"

            if step % self.config.validate_every_num_steps == 0:
                valid_metric = validate_by_finetuning_on_tasks(self, dataset, aml_run=aml_run, metric_to_use=metric_to_use)

                if aml_run:
                    # printing some measure of loss on all validation tasks.
                    if self.config.use_numeric_labels:
                        aml_run.log(f"valid_mean_r2", valid_metric)
                    else:
                        aml_run.log(f"valid_mean_avg_prec", valid_metric)

                if self.config.use_numeric_labels:
                    logger.info(
                        f"Validated at train step [{step}/{self.config.num_train_steps}],"
                        f" Valid R2: {valid_metric:.3f}",
                    )
                else:
                    logger.info(
                        f"Validated at train step [{step}/{self.config.num_train_steps}],"
                        f" Valid Avg. Prec.: {valid_metric:.3f}",
                    )

                # save model if validation avg prec is the best so far
                if valid_metric > best_validation_score:
                    best_validation_score = valid_metric
                    model_path = os.path.join(out_dir, "best_validation.pt")
                    self.save_model(model_path)
                    logger.info(f"Updated {model_path} to new best model at train step {step}")

        # save the fully trained model
        self.save_model(os.path.join(out_dir, "fully_trained.pt"))
