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
from fs_mol.models.dkt import DKTModel
from fs_mol.utils.dkt_utils import (
    DKTModelTrainer,
    evaluate_dkt_model,
)
from fs_mol.utils.test_utils import add_eval_cli_args, set_up_test_run


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test a DKT model on molecules.",
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
    parser.add_argument(
        "--test-time-adaptation",
        action="store_true",
        help="Turn on test time adaptation for DKT. Default: False.",
    )
    args = parser.parse_args()
    return args


def test(
    model: DKTModel,
    dataset: FSMolDataset,
    save_dir: str,
    context_sizes: List[int],
    num_samples: int,
    seed: int,
    batch_size: int,
):
    """
    Same procedure as validation for DKTModel. Each validation task is used to
    evaluate the model more than once, dependent on number of context sizes and samples.
    """

    return evaluate_dkt_model(
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
    out_dir, dataset = set_up_test_run("DKTModel", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=DKTModel,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        device=device,
    )

    model = DKTModelTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    )
    model.test_time_adaptation = args.test_time_adaptation
    if args.test_time_adaptation:
        scale = 0.25
        loc = np.log(model.gp_model.covar_module.base_kernel.lengthscale.item()) + scale**2
        lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        model.gp_model.covar_module.base_kernel.register_prior(
            "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
        )

        likelihood_noise = model.gp_likelihood.noise_covar.noise.item()
        scale = 0.25
        loc = np.log(likelihood_noise) + scale**2
        noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        model.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)

        model.mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp_likelihood, model.gp_model)

        model.save_gp_params()

    model.to(device)

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
