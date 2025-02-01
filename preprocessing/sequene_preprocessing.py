import re
import pandas as pd
import numpy as np

def categorize_sequences(
    df, 
    sequence_col='sequence', 
    output_col='sequence_category', 
    patterns=None
):
    """
    Categorizes peptide sequences based on regex patterns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        sequence_col (str): Column containing peptide sequences
        output_col (str): Output column for categories
        patterns (dict): Dictionary of {category: regex pattern}
    
    Returns:
        pd.DataFrame: Modified DataFrame with category column
    """
    if patterns is None:
        patterns = {
            "Standard": re.compile(r"^[A-Z]+$"),
            "Enantiomer Mix": re.compile(r"^[A-Za-z]+$"),
            "Chemical Formula": re.compile(r"^[A-Za-z0-9]+-[a-z]+$"),
            "Special Structure": re.compile(r"^[A-Za-z0-9\-]+-BLGTYTQDFNXFHTFPQTAIGVGAP$"),
            "Repeating Segments": re.compile(r"^[A-Za-z0-9\(\)\-]+-NH2$"),
            "Biotinylated": re.compile(r"\(biotin\)-[a-zA-Z0-9]+"),
            "N-terminal Acylated": re.compile(r"^Ac-[A-Z]+.*$"),
            "Cyclic": re.compile(r"^Cyclo\([A-Za-z0-9]+.*\)$"),
            "Amidated": re.compile(r"^.*-NH2$"),
            "Peptide with Substitutions": re.compile(r"[A-Za-z0-9\(\)\-\[\]]+"),
            "Uptake with Stearyl": re.compile(r".*(Stearyl|Myristoyl|Lauroyl)-[A-Za-z0-9]+"),
            "Polypeptide with Repeats": re.compile(r"poly-[A-Za-z0-9]+"),
            "Hexahistidine Tagged": re.compile(r"^(His6|HHHHHH)+.*"),
            "Unknown": re.compile(r".*")
        }

    def _categorizer(sequence):
        for category, pattern in patterns.items():
            if pattern.match(str(sequence)):
                return category
        return "Unknown"

    df[output_col] = df[sequence_col].apply(_categorizer)
    return df

def standardize_sequence(sequence):
    """
    Helper function to extract features from peptide sequences.
    Returns tuple of (cleaned_sequence, features...).
    """
    features = {
        'nh3_tail': False,
        'po3_pos': False,
        'biotinylated': False,
        'acylated_n_terminal': False,
        'cyclic': False,
        'amidated': False,
        'stearyl_uptake': False,
        'hexahistidine_tagged': False,
        'modifications': []
    }

    if not isinstance(sequence, str):
        return (np.nan, *features.values())

    # Feature detection
    features['nh3_tail'] = bool(re.search(r"-NH2$", sequence))
    features['po3_pos'] = bool(re.search(r"\(PO3\)", sequence))
    features['biotinylated'] = bool(re.search(r"\(biotin\)", sequence))
    features['acylated_n_terminal'] = bool(re.search(r"^Ac-", sequence))
    features['cyclic'] = bool(re.search(r"^Cyclo\(", sequence))
    features['amidated'] = bool(re.search(r"-NH2$", sequence))
    features['stearyl_uptake'] = bool(re.search(r"(Stearyl|Myristoyl|Lauroyl)", sequence))
    features['hexahistidine_tagged'] = bool(re.search(r"^(His6|HHHHHH)+", sequence))

    # Sequence cleaning
    cleaned = re.sub(r"[^A-Za-z]", "", sequence)
    cleaned_standard = re.sub(r"[^A-Z]", "", cleaned)
    
    if cleaned_standard.isalpha() and cleaned_standard.isupper():
        features['modifications'] = [
            (char.upper(), i+1) 
            for i, char in enumerate(cleaned) 
            if char.islower()
        ]
        return (cleaned_standard, *features.values())
    
    return (np.nan, *features.values())

def add_sequence_features(
    df,
    sequence_col='sequence',
    output_cols=[
        'standard_sequence', 'nh3_tail', 'po3_pos', 'biotinylated',
        'acylated_n_terminal', 'cyclic', 'amidated', 'stearyl_uptake',
        'hexahistidine_tagged', 'modifications'
    ]
):
    """
    Adds standardized sequence and feature columns to DataFrame.
    """
    df[output_cols] = df[sequence_col].apply(
        lambda x: pd.Series(standardize_sequence(x))
    )
    return df

def validate_sequences(df, sequence_col='standard_sequence'):
    """
    Filters valid sequences using standard amino acid check.
    """
    def _is_valid(seq):
        if not isinstance(seq, str):
            return False
        return bool(re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq))
    
    df[sequence_col] = df[sequence_col].apply(
        lambda s: s if _is_valid(s) else np.nan
    )
    return df