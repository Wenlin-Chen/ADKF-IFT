import sys
from pathlib import Path

sys.path.append("./chem_lib/models")
from chem_lib.models import adkf_trainer
from parser import get_args

sys.path.append("../")
from bayes_opt import bo_utils

# load the model
def load_model(model_weights_file: str, device: str):
    # model = bo_utils.ADKTModelFeatureExtractor(config=)
    # device = "cuda"  # TODO
    model = bo_utils.ADKTModelFeatureExtractor.build_from_model_file(
        model_weights_file,
        device=device
    ).to(device)
    return model

# def get_trainer(args, model):
#     return adkf_trainer.Meta_Trainer(args, model)


if __name__ == "__main__":
    args = get_args(root_dir=".")
    model_weights_file = "../outputs/FSMol_ADKTModel_gnn+ecfp+fc_2022-04-07_12-53-41/best_validation.pt" #2048 classification #model_weights_file = "../../outputs/FSMol_ADKTModel_gnn+ecfp+fc_2022-04-07_17-39-16/best_validation.pt" #2048 regression
    assert Path(model_weights_file).exists()

    # Model
    model = load_model(model_weights_file=model_weights_file, device=args.device)
    print(dir(model), type(model))

    # Trainer
    trainer = adkf_trainer.Meta_Trainer(args, model)

    # Testing
    best_avg_auc = trainer.test_step()