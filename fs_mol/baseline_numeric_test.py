#!/usr/bin/env python3
import json
import logging
import sys
from typing import Dict, Optional, List, Any

import numpy as np
import sklearn.ensemble
import sklearn.neighbors
from dpu_utils.utils import run_and_debug
from pyprojroot import here as project_root
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_task import FSMolTaskSample
from fs_mol.utils.cli_utils import str2bool
from fs_mol.utils.metrics import NumericEvalMetrics, compute_numeric_task_metrics
from fs_mol.utils.test_utils import (
    eval_model,
    add_eval_cli_args,
    set_up_test_run,
)

logger = logging.getLogger(__name__)

# TODO: extend to whichever models seem useful.
# hyperparam search params
DEFAULT_GRID_SEARCH: Dict[str, Dict[str, List[Any]]] = {
    "randomForestRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 20],
        "max_features": [None, "sqrt", "log2"],
        "min_samples_leaf": [2, 5],
    },
}

NAME_TO_MODEL_CLS: Dict[str, Any] = {
    "randomForestRegressor": sklearn.ensemble.RandomForestRegressor,
}


def test(
    model_name: str,
    task_sample: FSMolTaskSample,
    use_grid_search: bool = True,
    grid_search_parameters: Optional[Dict[str, Any]] = None,
    model_params: Dict[str, Any] = {},
) -> NumericEvalMetrics:
    train_data = task_sample.train_samples
    test_data = task_sample.test_samples

    # get data in to form for sklearn
    X_train = np.array([x.get_fingerprint() for x in train_data])
    X_test = np.array([x.get_fingerprint() for x in test_data])
    logger.info(f" Training {model_name} with {X_train.shape[0]} datapoints.")
    y_train = np.array([float(x.numeric_label) for x in train_data])
    y_test = np.array([float(x.numeric_label) for x in test_data])

    # apply log to the numeric label and standardize it
    log_y_train = np.log(y_train)
    standardize_mean = log_y_train.mean()
    standardize_std = log_y_train.std()
    log_y_train_standardized = (log_y_train - standardize_mean) / standardize_std
    log_y_test_standardized = (np.log(y_test) - standardize_mean) / standardize_std

    # use the train data to train a baseline model with CV grid search
    # reinstantiate model for each seed.
    if use_grid_search:
        if grid_search_parameters is None:
            grid_search_parameters = DEFAULT_GRID_SEARCH[model_name]
            # in the case of kNNs the grid search has to be modified -- one cannot have
            # more nearest neighbours than datapoints.
            grid_search = GridSearchCV(NAME_TO_MODEL_CLS[model_name](), grid_search_parameters)
        grid_search.fit(X_train, log_y_train_standardized)
        model = grid_search.best_estimator_
    else:
        model = NAME_TO_MODEL_CLS[model_name]()
        params = model.get_params()
        params.update(model_params)
        model.set_params()
        model.fit(X_train, log_y_train_standardized)

    # Compute test results:
    y_predicted_value = model.predict(X_test)
    test_metrics = compute_numeric_task_metrics(y_predicted_value, log_y_test_standardized)

    logger.info(f" Test metrics: {test_metrics}")

    return test_metrics


def run_from_args(args) -> None:
    out_dir, dataset = set_up_test_run(args.model, args)

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> NumericEvalMetrics:
        return test(
            model_name=args.model,
            task_sample=task_sample,
            use_grid_search=args.grid_search,
            model_params=args.model_params,
        )

    eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=args.train_sizes,
        out_dir=out_dir,
        num_samples=args.num_runs,
        seed=args.seed,
        filter_numeric_labels=True,
    )


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test sklearn models on tasks (numeric label).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        default="randomForestRegressor",
        choices=["randomForestRegressor"],
        help="The model to use.",
    )
    parser.add_argument(
        "--grid-search",
        type=str2bool,
        default=True,
        help="Perform grid search over hyperparameter space rather than use defaults/passed parameters.",
    )
    parser.add_argument(
        "--model-params",
        type=lambda s: json.loads(s),
        default={},
        help=(
            "JSON dictionary containing model hyperparameters, if not using grid search these will"
            " be used."
        ),
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")
    args = parser.parse_args()

    run_and_debug(lambda: run_from_args(args), args.debug)


if __name__ == "__main__":
    run()
