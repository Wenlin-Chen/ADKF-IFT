from rdkit import DataStructs
from rdkit.Chem import (
    MolFromSmiles,
    rdFingerprintGenerator,
)
import numpy as np


# The fingerprints provided in FS-Mol are count-based,
# which cannot be used as inputs to the Tanimoto kernel.
# This function returns a list of binary fingerprints (using count simulation by default).
def get_binary_fingerprints(data, use_count_simulation=True):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, useCountSimulation=use_count_simulation)
    fp_list = []

    for x in data:
        rdkit_mol = MolFromSmiles(x.smiles)
        fp_vec = fpgen.GetFingerprint(rdkit_mol)
        fp_numpy = np.zeros((0,), np.int8) 
        DataStructs.ConvertToNumpyArray(fp_vec, fp_numpy)
        fp_list.append(fp_numpy)

    return fp_list