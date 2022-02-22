import argparse
import logging
import sys
from typing import List

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import FSMolDataset
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.adaptive_dkt import ADKTModel
from fs_mol.utils.adaptive_dkt_utils import (
    ADKTModelTrainer,
    evaluate_adkt_model,
)
from fs_mol.utils.test_utils import add_walltime_cli_args, set_up_test_run

import time


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test an Adaptive DKT model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "TRAINED_MODEL",
        type=str,
        help="File to load model from (determines model architecture).",
    )

    add_walltime_cli_args(parser)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="Maximum batch size to allow when running through inference on model.",
    )
    parser.add_argument(
        "--use-fresh-param-init",
        action="store_true",
        help="Do not use trained weights, but start from a fresh, random initialisation.",
    )
    args = parser.parse_args()
    return args


def test(
    model: ADKTModel,
    dataset: FSMolDataset,
    save_dir: str,
    context_sizes: List[int],
    num_samples: int,
    seed: int,
    batch_size: int,
):
    """
    Same procedure as validation for ADKTModel. Each validation task is used to
    evaluate the model more than once, dependent on number of context sizes and samples.
    """

    return evaluate_adkt_model(
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
    out_dir, dataset = set_up_test_run("walltime_ADKTModel", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=ADKTModel,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        device=device,
    )

    model = ADKTModelTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    )

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    test(
        model,
        dataset,
        save_dir=out_dir,
        context_sizes=args.train_sizes,
        num_samples=args.num_runs,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    walltime = end_time - start_time
    logger.info(f"Walltime: {walltime} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
