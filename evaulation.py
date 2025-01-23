from rdkit import Chem
from rdkit.Chem import AllChem
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import kendalltau
from typing import Tuple


def mol_has_3D(mol):
    try:
        mol.GetConformer()
    except ValueError:
        raise ValueError("Cannot get conformer for molecule, is likely 2D")

    if not mol.GetConformer().Is3D():
        raise ValueError("Molecule is not 3D")


def eval_poses(
    preds: list[Chem.Mol], refs: list[Chem.Mol], cutoff=2.0
) -> Tuple[np.ndarray, float]:
    """
    Evaluate the poses of the predicted molecules against the reference molecules.

    Calculates the % correct with respect to the RMSD cutoff value.

    Parameters
    ----------
    preds : list[Chem.Mol]
        List of predicted molecules.
    refs : list[Chem.Mol]

    cutoff : float
        The cutoff value for the RMSD value. Default is 2.0.

    Returns
    -------
    Tuple[np.ndarray, float]
        Returns a tuple of the RMSD values and the percentage of RMSD values less than the cutoff

    """
    if len(preds) != len(refs):
        raise ValueError("mismatched lengths in preds vs references")

    # find symm corrected  heavy atom RMSDs
    rmsds = []
    for pred, ref in zip(preds, refs):
        mol_has_3D(pred)
        mol_has_3D(ref)
        if pred.GetNumHeavyAtoms() != ref.GetNumHeavyAtoms():
            raise ValueError("mismatched number of atoms")
        pred_spy = Molecule.from_rdkit(pred)
        ref_spy = Molecule.from_rdkit(ref)
        rmsd = rmsdwrapper(ref_spy, pred_spy, symmetry=True, strip=True)
        rmsds.extend(rmsd)

    rmsds = np.asarray(rmsds)

    # calculate % less than cutoff
    mask = rmsds <= cutoff

    correct = sum(mask)
    prob = (correct / rmsds.shape[0]) * 100

    collect = {}
    collect["raw_rmsds"] = rmsds
    collect["percentage_correct"] = prob

    return collect


def eval_potency(
    preds: dict[str, list], refs: dict[str, list]
) -> Tuple[dict[str, float]]:
    """
    Eval pIC50 potency performance with MAE (already log10 transformed)

    Parameters
    ----------
    preds : dict[str, list]
        Dictionary of predicted pIC50 values for SARS-CoV-2 and MERS-CoV Mpro.
    refs : dict[str, list]
        Dictionary of reference pIC50 values for SARS-CoV-2 and MERS-CoV Mpro.

    Returns
    -------
    dict[str, float]
        Returns a dictonary of summary statistics
    """
    keys = {"pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"}
    collect = {}
    for k in keys:
        if k not in preds.keys() or k not in refs.keys():
            raise ValueError("required key not present")
        pred = preds[k]
        ref = refs[k]
        mae = mean_absolute_error(ref, pred)
        ktau = kendalltau(ref, pred)
        r2 = r2_score(ref, pred)

        # subchallenge statistics
        collect[k + " MAE"] = mae
        collect[k + " ktau"] = ktau.statistic
        collect[k + " R2"] = r2

    # compute macro average MAE
    mae_entries = {k for k in collect.keys() if "MAE" in k}
    macro_mae = np.mean([collect[k] for k in mae_entries])
    collect["pIC50_macro_mae_avg"] = macro_mae

    # compute macro average R2
    R2_entries = {k for k in collect.keys() if "R2" in k}
    macro_R2 = np.mean([collect[k] for k in R2_entries])
    collect["pIC50_macro_R2_avg"] = macro_R2

    return collect


def eval_admet(
    preds: dict[str, list], refs: dict[str, list]
) -> Tuple[dict[str, float], np.ndarray]:
    """
    Eval ADMET targets with MAE for pre-log10 transformed targets (LogD) and MALE  (MAE on log10 transformed dataset) on non-transformed data

    This provides a "relative" error metric that will not be as sensitive to the large outliers with huge errors. This is sometimes known as MALE.

    Parameters
    ----------
    preds : dict[str, list]
        Dictionary of predicted ADMET values.
    refs : dict[str, list]
        Dictionary of reference ADMET values.

    Returns
    -------
    dict[str, float]
        Returns a dictonary of summary statistics
    """
    keys = {
        "MLM",
        "HLM",
        "KSOL",
        "LogD",
        "MDR1-MDCKII",
    }
    # will be treated as is
    logscale_endpts = {"LogD"}

    collect = {}
    for k in keys:
        if k not in preds.keys() or k not in refs.keys():
            raise ValueError("required key not present")

        pred = preds[k]
        ref = refs[k]

        if k in logscale_endpts:
            # already log10scaled
            mae = mean_absolute_error(ref, pred)
            r2 = r2_score(ref, pred)
        else:
            # clip to a detection limit
            epsilon = 1e-8
            pred = np.clip(pred, a_min=epsilon, a_max=None)
            ref = np.clip(ref, a_min=epsilon, a_max=None)

            # transform both log10scale
            pred_log10s = np.log10(pred)
            ref_log10s = np.log10(ref)

            # compute MALE and R2 in log space
            mae = mean_absolute_error(ref_log10s, pred_log10s)
            r2 = r2_score(ref_log10s, pred_log10s)

        collect[k + " MAE"] = mae
        collect[k + " R2"] = r2

    # compute macro average MAE
    mae_entries = {k for k in collect.keys() if "MAE" in k}
    macro_mae = np.mean([collect[k] for k in mae_entries])
    collect["ADMET_macro_mae_avg"] = macro_mae

    # compute macro average R2
    R2_entries = {k for k in collect.keys() if "R2" in k}
    macro_R2 = np.mean([collect[k] for k in R2_entries])
    collect["ADMET_macro_R2_avg"] = macro_R2

    return collect
