import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PARModelTrainerConfig(PARModelConfig):
    batch_size: int = 256
    tasks_per_batch: int = 16
    support_set_size: int = 16
    query_set_size: int = 256

    num_train_steps: int = 10000
    validate_every_num_steps: int = 50
    validation_support_set_sizes: Tuple[int] = (16, 128)
    validation_query_set_size: int = 256
    validation_num_samples: int = 5

    outer_learning_rate: float = 0.001
    clip_value: Optional[float] = None

    use_numeric_labels: bool = False

def get_predictions(model, data_batch: DKTBatch, train=True, **kwargs):
    """Make PAR predictions"""

    # if train:
    s_logits, q_logits, adj, s_node_emb = model(input_batch=data_batch, **kwargs)
    pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj,}

    return pred_dict

def get_loss():
    """Loss for the PAR model"""
    # TODO: need to implement this based on "get_loss"
    raise NotImplementedError
    

def run_on_batches(
    model: PARModel,
    batches: List[DKTBatch],
    batch_labels: List[torch.Tensor],
    batch_numeric_labels: List[torch.Tensor],
    train: bool = False,
    #tasks_per_batch: int = 1,
):
    """Does inner loop adaptation."""

    # TODO: this function requires calls to "get_loss" and "get_predictions" to be implemented
    # TODO: also requires MAML-wrapped model: probably we should do this in this function?

    if train:
        assert len(batches) == 1

    total_loss, total_num_samples = 0.0, 0
    task_preds: List[np.ndarray] = []
    task_labels: List[np.ndarray] = []

    #num_gradient_accumulation_steps = len(batches) * tasks_per_batch
    cloned_models = []
    for batch_features, this_batch_labels, this_batch_numeric_labels in zip(batches, batch_labels, batch_numeric_labels):
        
        # MAML clone
        # TODO: MAML wrap here??
        model = self.model.clone()
        model.train()
        adaptable_weights = self.get_adaptable_weights(model)
        cloned_models.append(model)
                        
        # MAML adaptation
        for inner_step in range(self.inner_update_step):
            pred_adapt = self.get_prediction(model, batch_features, train=True)
            loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
            model.adapt(loss_adapt, adaptable_weights = adaptable_weights)
        
        # Compute loss on the support set at test time
        if not train:
            model.eval()
            with torch.no_grad():
                pred_eval = self.get_prediction(model, eval_data, train=False)

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
        self.optimizer = torch.optim.Adam(self.feature_extractor_params(), config.learning_rate)
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

        # Overall outer loop steps
        for step in range(1, self.config.num_train_steps + 1):
            
            # Number of repeated steps on this batch
            for update_on_current_batch_idx in range(self.update_step):

                # Do inner loop on batch, one task at a time
                pred_losses = []
                for _ in range(self.config.tasks_per_batch):
                    task_sample = next(train_task_sample_iterator)
                    train_task_sample = torchify(task_sample, device=device)
                    batches = train_task_sample.batches
                    batch_labels = train_task_sample.batch_labels
                    batch_numeric_labels = train_task_sample.batch_numeric_labels
                    cloned_models, _ = run_on_batches(
                        self,
                        batches=batches,
                        batch_labels=batch_labels,
                        batch_numeric_labels=batch_numeric_labels,
                        train=True,
                        #tasks_per_batch=self.config.tasks_per_batch,
                    )

                    # Make model predictions on query set
                    assert len(batches) == len(cloned_models) == 1
                    pred_eval = get_prediction(model, batches[0], train=True)
                    pred_loss = self.get_loss(model, batches[0], pred_eval, train=True, flag = 0)
                    assert False  # TODO: normalize loss
                    pred_losses.append(pred_loss)
                    del pred_loss
                
                # Outer loop update
                pred_losses_tensor = torch.stack(pred_losses)
                overall_loss = torch.sum(pred_losses_tensor) / len(pred_losses)
                self.optimizer.zero_grad()
                overall_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())

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
