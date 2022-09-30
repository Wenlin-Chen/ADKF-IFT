import argparse
import logging
import sys
from typing import List

import torch
import numpy as np
import gpytorch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import FSMolDataset
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.dkl import DKLModel
from fs_mol.utils.dkl_utils import (
    DKLModelTrainer,
    DKLModelTrainerConfig,
    evaluate_dkl_model,
)
from fs_mol.modules.graph_feature_extractor import (
    add_graph_feature_extractor_arguments,
    make_graph_feature_extractor_config_from_args,
)
from fs_mol.utils.test_utils import add_eval_cli_args, set_up_test_run


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test a DKL model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "gnn",
            "ecfp",
            "pc-descs",
            "ecfp+fc",
            "pc-descs+fc",
            "gnn+ecfp+fc",
            "gnn+ecfp+pc-descs+fc",
        ],
        default="gnn+ecfp+fc",
        help="Choice of features to use",
    )
    add_graph_feature_extractor_arguments(parser)

    parser.add_argument(
        "--num_train_steps", type=int, default=50, help="Number of training steps."
    )
    
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--clip_value", type=float, default=1.0, help="Gradient norm clipping value"
    )
    parser.add_argument(
        "--use-ard",
        action="store_true",
        help="Use a different lengthscale for each input dimension to the GP.",
    )
    parser.add_argument(
        "--gp-kernel",
        type=str,
        default="matern",
        help="The GP kernel.",
    )
    parser.add_argument(
        "--use-lengthscale-prior",
        action="store_true",
        help="Put a logNormal prior over the lengthscale(s).",
    )
    parser.add_argument(
        "--use-numeric-labels",
        action="store_true",
        help="Perform regression for the numeric labels (log concentration). Default: perform binary classification for the bool labels (active/inactive).",
    )


    add_eval_cli_args(parser)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="Maximum batch size to allow when running through inference on model.",
    )
    args = parser.parse_args()
    return args


def make_trainer_config(args: argparse.Namespace) -> DKLModelTrainerConfig:
    return DKLModelTrainerConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        used_features=args.features,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        learning_rate=args.lr,
        clip_value=args.clip_value,
        use_ard=args.use_ard,
        gp_kernel=args.gp_kernel,
        use_lengthscale_prior=args.use_lengthscale_prior,
        use_numeric_labels=args.use_numeric_labels,
    )


def test(
    model: DKLModel,
    dataset: FSMolDataset,
    save_dir: str,
    context_sizes: List[int],
    num_samples: int,
    seed: int,
    batch_size: int,
):
    """
    Same procedure as validation for DKLModel. Each validation task is used to
    evaluate the model more than once, dependent on number of context sizes and samples.
    """

    return evaluate_dkl_model(
        model,
        dataset,
        support_sizes=context_sizes,
        num_samples=num_samples,
        seed=seed,
        batch_size=batch_size,
        save_dir=save_dir,
    )


def main():
    args = parse_command_line()
    config = make_trainer_config(args)
    out_dir, dataset = set_up_test_run("DKLModel", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DKLModelTrainer(config=config).to(device)

    logger.info(f"\tNum parameters {sum(p.numel() for p in model.parameters())}")
    logger.info(f"\tModel:\n{model}")

    test(
        model,
        dataset,
        save_dir=out_dir,
        context_sizes=args.train_sizes,
        num_samples=args.num_runs,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
