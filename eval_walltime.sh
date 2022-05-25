#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate fsmol

CUDA_VISIBLE_DEVICES=1, python fs_mol/cnp_walltime.py outputs/FSMol_CNPModel_gnn+ecfp+fc_2022-04-11_16-46-27/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/cnp_walltime.py outputs/FSMol_CNPModel_gnn+ecfp+fc_2022-04-11_16-46-27/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/cnp_walltime.py outputs/FSMol_CNPModel_gnn+ecfp+fc_2022-04-11_16-46-27/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/cnp_walltime.py outputs/FSMol_CNPModel_gnn+ecfp+fc_2022-04-11_16-46-27/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/cnp_walltime.py outputs/FSMol_CNPModel_gnn+ecfp+fc_2022-04-11_16-46-27/best_validation.pt ../fs-mol-dataset/

CUDA_VISIBLE_DEVICES=1, python fs_mol/dkt_walltime.py outputs/FSMol_DKTModel_gnn+ecfp+fc_2022-04-08_11-01-28/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/dkt_walltime.py outputs/FSMol_DKTModel_gnn+ecfp+fc_2022-04-08_11-01-28/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/dkt_walltime.py outputs/FSMol_DKTModel_gnn+ecfp+fc_2022-04-08_11-01-28/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/dkt_walltime.py outputs/FSMol_DKTModel_gnn+ecfp+fc_2022-04-08_11-01-28/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/dkt_walltime.py outputs/FSMol_DKTModel_gnn+ecfp+fc_2022-04-08_11-01-28/best_validation.pt ../fs-mol-dataset/

CUDA_VISIBLE_DEVICES=1, python fs_mol/protonet_walltime.py ../fs-mol-checkpoints/PNSupport64_best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/protonet_walltime.py ../fs-mol-checkpoints/PNSupport64_best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/protonet_walltime.py ../fs-mol-checkpoints/PNSupport64_best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/protonet_walltime.py ../fs-mol-checkpoints/PNSupport64_best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/protonet_walltime.py ../fs-mol-checkpoints/PNSupport64_best_validation.pt ../fs-mol-dataset/

CUDA_VISIBLE_DEVICES=1, python fs_mol/adaptive_dkt_walltime.py outputs/FSMol_ADKTModel_gnn+ecfp+fc_2022-04-08_10-59-10/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/adaptive_dkt_walltime.py outputs/FSMol_ADKTModel_gnn+ecfp+fc_2022-04-08_10-59-10/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/adaptive_dkt_walltime.py outputs/FSMol_ADKTModel_gnn+ecfp+fc_2022-04-08_10-59-10/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/adaptive_dkt_walltime.py outputs/FSMol_ADKTModel_gnn+ecfp+fc_2022-04-08_10-59-10/best_validation.pt ../fs-mol-dataset/
CUDA_VISIBLE_DEVICES=1, python fs_mol/adaptive_dkt_walltime.py outputs/FSMol_ADKTModel_gnn+ecfp+fc_2022-04-08_10-59-10/best_validation.pt ../fs-mol-dataset/

CUDA_VISIBLE_DEVICES=1, python fs_mol/maml_walltime.py ../fs-mol-dataset/ --trained-model ../fs-mol-checkpoints/MAMLSupport16_best_validation.pkl
CUDA_VISIBLE_DEVICES=1, python fs_mol/maml_walltime.py ../fs-mol-dataset/ --trained-model ../fs-mol-checkpoints/MAMLSupport16_best_validation.pkl
CUDA_VISIBLE_DEVICES=1, python fs_mol/maml_walltime.py ../fs-mol-dataset/ --trained-model ../fs-mol-checkpoints/MAMLSupport16_best_validation.pkl
CUDA_VISIBLE_DEVICES=1, python fs_mol/maml_walltime.py ../fs-mol-dataset/ --trained-model ../fs-mol-checkpoints/MAMLSupport16_best_validation.pkl
CUDA_VISIBLE_DEVICES=1, python fs_mol/maml_walltime.py ../fs-mol-dataset/ --trained-model ../fs-mol-checkpoints/MAMLSupport16_best_validation.pkl