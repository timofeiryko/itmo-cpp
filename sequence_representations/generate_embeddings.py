import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

from blomap import blomap_extra_encode

def get_morgan_fingerprint(smiles, radius=2, nBits=1024):
    """Generate a Morgan fingerprint from a SMILES string."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
        return np.array(fingerprint)
    except Exception:
        global counter
        counter += 1
        return np.zeros(nBits)  # Return a zero array of the desired shape if the SMILES is invalid

# Blomap-related functions
def add_padding(sequence: str, max_length: int, pad_char: str = 'X') -> str:
    """Pad sequence to specified length with given padding character"""
    return sequence + pad_char * (max_length - len(sequence)) if isinstance(sequence, str) else sequence

def calculate_max_sequence_length(df: pd.DataFrame, seq_col: str = 'sequence') -> int:
    """Calculate maximum sequence length in dataframe"""
    return df[seq_col].apply(lambda x: len(x) if isinstance(x, str) else 0).max()

def generate_blomap_embeddings(
    df: pd.DataFrame, 
    seq_col: str = 'standard_sequence',
    blomap_encoder: callable = blomap_extra_encode
) -> np.ndarray:
    """Generate Blomap embeddings for sequences"""
    max_len = calculate_max_sequence_length(df, seq_col)
    df['adjusted_sequence'] = df[seq_col].apply(
        lambda s: add_padding(s, max_len) if isinstance(s, str) else s
    )
    
    embeddings = np.array([
        blomap_encoder(seq) if isinstance(seq, str) else np.nan 
        for seq in df['adjusted_sequence']
    ])
    
    return embeddings[~np.isnan(embeddings).any(axis=1)]

# ProtBERT-related functions
class ProtBERTEmbedder:
    """Class for handling ProtBERT embeddings generation"""
    
    def __init__(self, model_name: str = "Rostlab/prot_bert_bfd"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Embed single sequence"""
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to device
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    def embed_batch(self, sequences: list, batch_size: int = 32) -> np.ndarray:
        """Embed batch of sequences with progress bar"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="ProtBERT Batches"):
            batch = sequences[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Correct device transfer
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)

# Data processing functions
def process_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequence length and padding columns"""
    df['seq_length'] = df['sequence'].apply(lambda s: len(s) if isinstance(s, str) else 0)
    max_len = df['seq_length'].max()
    df['adjusted_sequence'] = df['standard_sequence'].apply(
        lambda s: add_padding(s, max_len) if isinstance(s, str) else s
    )
    return df

if __name__ == "__main__":

    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Generate protein sequence embeddings')
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    GLOBAL_LEN = len(df)
    print("Number of sequences: ", GLOBAL_LEN)
    processed_df = process_sequences(df)

    print("Generating Blomap embeddings...")

    SEQ_LEN = calculate_max_sequence_length(processed_df, 'adjusted_sequence')
    df['adjusted_sequence'] = df['standard_sequence'].apply(lambda s: add_padding(s, SEQ_LEN) if isinstance(s, str) else s)

    # Get a non-NaN sequence
    non_nan_sequence = df['adjusted_sequence'].dropna().iloc[0]

    # Get the Blomap embedding for the sequence
    embedding = blomap_extra_encode(non_nan_sequence)

    # Get the length of the embedding
    SHAPE = len(embedding)
    print("Shape of Blomap embedding: ", SHAPE)

    blomap_embeddings = list(df['adjusted_sequence'].apply(blomap_extra_encode, args=(SHAPE,)))
    blomap_embeddings = np.array(blomap_embeddings)

    assert blomap_embeddings.shape[0] == GLOBAL_LEN

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(SCRIPT_DIR, 'blomap_embeddings.npy')
    np.save(save_path, blomap_embeddings)
    print(f"Saved Blomap embeddings with shape {blomap_embeddings.shape} to:\n{save_path}")

    print("Generating ProtBERT embeddings...")

    protbert_embedder = ProtBERTEmbedder()

    # Use original sequences (unpadded) and handle NaNs
    sequences = df['standard_sequence'].fillna('').tolist()  # Fill NaN with empty strings

    # Generate ProtBERT embeddings
    protbert_embeddings = protbert_embedder.embed_batch(sequences, batch_size=32)
    assert protbert_embeddings.shape[0] == GLOBAL_LEN

    save_path = os.path.join(SCRIPT_DIR, 'protbert_embeddings.npy')
    np.save(save_path, protbert_embeddings)
    print(f"Saved ProtBERT embeddings with shape {protbert_embeddings.shape} to:\n{save_path}")

    # Generate Morgan fingerprints

    print("Generating Morgan fingerprints...")
    counter = 0
    fingerprints = []
    for smiles in tqdm(df['smiles_sequence']):
        fingerprint = get_morgan_fingerprint(smiles)
        fingerprints.append(fingerprint)
    fingerprints_array = np.array(fingerprints)

    assert fingerprints_array.shape[0] == GLOBAL_LEN

    save_path = os.path.join(SCRIPT_DIR, 'morgan_fingerprints.npy')
    np.save(save_path, fingerprints_array)
    print(f"Saved Morgan fingerprints with shape {fingerprints_array.shape} to:\n{save_path}")

    print("All representations have been generated successfully!")


