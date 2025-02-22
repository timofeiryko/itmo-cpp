import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os

# ======== Descriptors
def compute_sequence_descriptors(seq):
    """aa sequence descriptors"""
    try:
        seq = seq.upper()
        analyzer = ProteinAnalysis(seq)
        mw = analyzer.molecular_weight()
        gravy = analyzer.gravy()
        pi = analyzer.isoelectric_point()
        charge = analyzer.charge_at_pH(7.4)
        aromaticity = analyzer.aromaticity()
        flexibility = np.mean(analyzer.flexibility()) if analyzer.flexibility() else np.nan
        aliphatic_index = sum(analyzer.count_amino_acids().get(aa, 0) for aa in "AVILMF") / len(seq)
        boman_index = sum(analyzer.secondary_structure_fraction())
        charge_density = charge / len(seq)
        hydrophobic_aa = sum(seq.count(aa) for aa in "AVILMFYW")
        polar_aa = sum(seq.count(aa) for aa in "STNQ")
        positive_aa = sum(seq.count(aa) for aa in "KRH")
        negative_aa = sum(seq.count(aa) for aa in "DE")

        return [mw, gravy, pi, charge, charge_density, aromaticity, flexibility, aliphatic_index, boman_index,
                hydrophobic_aa, polar_aa, positive_aa, negative_aa]
    except Exception as e:
        print(f"mistake seq {seq}: {e}")
        return [np.nan] * 13

def compute_chemical_descriptors(smiles):
    """chem descriptors SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * 8

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)
        fsp3 = Descriptors.FractionCSP3(mol)

        return [mw, logp, tpsa, hbd, hba, rot_bonds, rings, fsp3]
    except Exception as e:
        print(f"mistake SMILES {smiles}: {e}")
        return [np.nan] * 8

# =======main
if __name__ == "__main__":
    input_file = "for_regr.csv"  
    output_file = "for_regr_descriptors.csv"  

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"file {input_file} not found")

    df4 = pd.read_csv(input_file)

    if 'standard_sequence' not in df4.columns or 'smiles_sequence' not in df4.columns:
        raise ValueError("file must have 'standard_sequence' and 'smiles_sequence' columns")

    df4 = df4.dropna(subset=['standard_sequence', 'smiles_sequence'])  

    print("computing sequence descriptors...")
    sequence_descriptors = df4['standard_sequence'].apply(compute_sequence_descriptors)

    print("computing chemical descriptors...")
    chemical_descriptors = df4['smiles_sequence'].apply(compute_chemical_descriptors)
    
    seq_desc_names = ['MW', 'GRAVY', 'pI', 'Charge', 'Charge_Density', 'Aromaticity', 'Flexibility',
                      'Aliphatic_Index', 'Boman_Index', 'Hydrophobic_AA', 'Polar_AA', 'Positive_AA', 'Negative_AA']
    chem_desc_names = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'Fsp3']

    df_seq_desc = pd.DataFrame(sequence_descriptors.tolist(), columns=seq_desc_names, index=df4.index)
    df_chem_desc = pd.DataFrame(chemical_descriptors.tolist(), columns=chem_desc_names, index=df4.index)

    df4 = pd.concat([df4, df_seq_desc, df_chem_desc], axis=1)

    df4.to_csv(output_file, index=False)
    print(f"file saved as {output_file}")