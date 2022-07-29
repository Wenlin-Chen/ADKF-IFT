import argparse
import logging
import sys
from typing import List

import torch
from pyprojroot import here as project_root

from pathlib import Path
sys.path.insert(0, str(project_root()))
PAR_CODE_PATH = "./PAR-NeurIPS21"
assert Path(PAR_CODE_PATH).exists()
sys.path.insert(0, PAR_CODE_PATH)

from fs_mol.data import FSMolDataset
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.par import PARModel
from fs_mol.utils.par_utils import (
    PARModelTrainer,
    evaluate_par_model,
)
from fs_mol.utils.test_utils import add_eval_cli_args, set_up_test_run

from chem_lib.models.maml import MAML


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test a PAR model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "TRAINED_MODEL",
        type=str,
        help="File to load model from (determines model architecture).",
    )

    add_eval_cli_args(parser)

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
    model: PARModel,
    dataset: FSMolDataset,
    save_dir: str,
    context_sizes: List[int],
    num_samples: int,
    seed: int,
    batch_size: int,
):
    """
    Same procedure as validation for PARModel. Each validation task is used to
    evaluate the model more than once, dependent on number of context sizes and samples.
    """

    return evaluate_par_model(
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
    out_dir, dataset = set_up_test_run("PARModel", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=PARModel,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        device=device,
    )

    model = PARModelTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    ).to(device)

    maml_model = MAML(model, lr=model.config.inner_learning_rate, first_order=not model.config.second_order_maml, anil=False, allow_unused=True)

    test(
        maml_model,
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
