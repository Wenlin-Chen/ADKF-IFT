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
from fs_mol.models.dkl import DKLModel, DKLModelConfig
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


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DKLModelTrainerConfig(DKLModelConfig):
    batch_size: int = 256

    num_train_steps: int = 100

    learning_rate: float = 0.001
    clip_value: Optional[float] = None

    use_ard: bool = False
    gp_kernel: str = "matern"
    use_lengthscale_prior: bool = False
    use_numeric_labels: bool = False


def run_on_batches(
    model: DKLModel,
    batches: List[DKTBatch],
    batch_labels: List[torch.Tensor],
    batch_numeric_labels: List[torch.Tensor],
    train: bool = False,
):

    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_num_samples = 0.0, 0
    task_preds: List[np.ndarray] = []
    task_labels: List[np.ndarray] = []

    for batch_features, this_batch_labels, this_batch_numeric_labels in zip(batches, batch_labels, batch_numeric_labels):

        # Compute loss at training time
        if train:
            model.load_state_dict(model.init_params)
            torch.set_grad_enabled(True)

            for i in range(model.config.num_train_steps):
                model.optimizer.zero_grad()
                # Compute task loss
                batch_logits = model(batch_features, train=True)
                batch_loss = model.compute_loss(batch_logits)
                batch_loss.backward()
                if model.config.clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.clip_value)
                model.optimizer.step()
                if model.lr_scheduler is not None:
                    model.lr_scheduler.step()

            per_sample_loss = batch_loss.detach()
            break

        # compute metric at test time
        else:
            # Compute task loss
            batch_logits = model(batch_features, train=False)
            with torch.no_grad():
                if model.config.use_numeric_labels:
                    batch_preds = batch_logits.mean.detach().cpu().numpy()
                    task_labels.append(this_batch_numeric_labels.detach().cpu().numpy())
                else:
                    batch_preds = torch.sigmoid(batch_logits.mean).detach().cpu().numpy()
                    task_labels.append(this_batch_labels.detach().cpu().numpy())
                task_preds.append(batch_preds)

    if train:
        # we will report loss per sample as before.
        metrics = None
    else:
        per_sample_loss = None

        predictions = np.concatenate(task_preds, axis=0)
        labels=np.concatenate(task_labels, axis=0)
        if model.config.use_numeric_labels:
            metrics = compute_numeric_task_metrics(predictions=predictions, labels=labels)
        else:
            metrics = compute_binary_task_metrics(predictions=predictions, labels=labels)

    return per_sample_loss, metrics


def evaluate_dkl_model(
    model: DKLModel,
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

        _none1, _none2 = run_on_batches(
            model,
            batches=dkt_task_sample.batches,
            batch_labels=dkt_task_sample.batch_labels,
            batch_numeric_labels=dkt_task_sample.batch_numeric_labels,
            train=True,
        )

        _, result_metrics = run_on_batches(
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


class DKLModelTrainer(DKLModel):
    def __init__(self, config: DKLModelTrainerConfig):
        super().__init__(config)
        self.config = config
        self.optimizer = torch.optim.Adam(self.parameters(), config.learning_rate)
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

