#!/usr/bin/env python3
import json
import logging
import sys
from typing import Dict, Optional, List, Any

import numpy as np
import torch
import gpytorch
from dpu_utils.utils import run_and_debug
from pyprojroot import here as project_root

from botorch.optim.fit import fit_gpytorch_scipy

sys.path.insert(0, str(project_root()))

from fs_mol.utils.gp_utils import ExactTanimotoGP

from fs_mol.data.fsmol_task import FSMolTaskSample
from fs_mol.utils.metrics import compute_numeric_task_metrics
from fs_mol.utils.test_utils import (
    eval_model,
    add_eval_cli_args,
    set_up_test_run,
)
from fs_mol.utils.metrics import compute_binary_task_metrics
from fs_mol.utils.test_utils import (
    eval_model,
    add_eval_cli_args,
    set_up_test_run,
)

logger = logging.getLogger(__name__)


def test(
    model_name: str,
    task_sample: FSMolTaskSample,
    device: torch.device,
    use_numeric_labels: bool,
):
    train_data = task_sample.train_samples
    test_data = task_sample.test_samples

    # get data in to form for sklearn
    X_train = torch.from_numpy(np.array([x.get_fingerprint() for x in train_data])).float().to(device)
    X_test = torch.from_numpy(np.array([x.get_fingerprint() for x in test_data])).float().to(device)
    logger.info(f" Training {model_name} with {X_train.shape[0]} datapoints.")
    
    if use_numeric_labels:
        y_train = torch.FloatTensor([float(x.numeric_label) for x in train_data])
        y_test = torch.FloatTensor([float(x.numeric_label) for x in test_data])

        # apply log to the numeric label and standardize it
        log_y_train = torch.log(y_train)
        standardize_mean = log_y_train.mean()
        standardize_std = log_y_train.std()
        y_train_converted = (log_y_train - standardize_mean) / standardize_std
        y_test_converted = (torch.log(y_test) - standardize_mean) / standardize_std
    
    else:
        y_train = torch.FloatTensor([float(x.bool_label) for x in train_data]).to(device)
        y_test = np.array([float(x.bool_label) for x in test_data])
        y_train_converted = y_train * 2.0 - 1.0

    # define model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactTanimotoGP(X_train, y_train_converted, likelihood, use_numeric_labels).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)

    # fit model
    model.train()
    likelihood.train()
    fit_gpytorch_scipy(mll)

    # Compute test results:
    model.eval()
    likelihood.eval()

    test_batch_size = 700
    n_test = X_test.shape[0]
    n_batches = n_test // test_batch_size
    n_rest = n_test % test_batch_size

    task_preds: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(n_batches+1):
            if i < n_batches:
                idx_start = i * test_batch_size
                idx_end = (i+1) * test_batch_size
            else:
                idx_start = n_batches * test_batch_size
                idx_end = n_batches * test_batch_size + n_rest

            batch_X_test = X_test[idx_start: idx_end].to(device)
            batch_pred_dist = likelihood(model(batch_X_test))
            if use_numeric_labels:
                batch_preds = batch_pred_dist.mean.detach().cpu().numpy()
            else:
                batch_preds = torch.sigmoid(batch_pred_dist.mean).detach().cpu().numpy()
            task_preds.append(batch_preds)

    y_predicted_value = np.concatenate(task_preds, axis=0)
    if use_numeric_labels:
        test_metrics = compute_numeric_task_metrics(predictions=y_predicted_value, labels=y_test_converted.numpy())
    else:
        test_metrics = compute_binary_task_metrics(predictions=y_predicted_value, labels=y_test)

    logger.info(f" Test metrics: {test_metrics}")

    return test_metrics


def run_from_args(args) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\tDevice: {device}")

    out_dir, dataset = set_up_test_run("GPST", args)

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ):
        return test(
            model_name="GPST",
            task_sample=task_sample,
            device=device,
            use_numeric_labels=args.use_numeric_labels,
        )

    eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=args.train_sizes,
        out_dir=out_dir,
        num_samples=args.num_runs,
        seed=args.seed,
        filter_numeric_labels=args.use_numeric_labels,
    )


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test GPST models on tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--use-numeric-labels",
        action="store_true",
        help="Perform regression for the numeric labels (log concentration). Default: perform binary classification for the bool labels (active/inactive).",
    )

    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")
    args = parser.parse_args()

    run_and_debug(lambda: run_from_args(args), args.debug)


if __name__ == "__main__":
    run()
