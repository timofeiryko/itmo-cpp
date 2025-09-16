import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


df = pd.read_csv("for_regr_with_descrip.csv")

# ======== ProtBERT Embeddings ========
model_name = "Rostlab/prot_bert_bfd"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_protbert_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

sequences = df["standard_sequence"].fillna("").tolist()
batch_size = 32  
protbert_embeddings = []
for i in range(0, len(sequences), batch_size):
    batch_sequences = sequences[i:i + batch_size]
    inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    protbert_embeddings.append(batch_embeddings)

protbert_embeddings = np.vstack(protbert_embeddings)
np.save("protbert_embeddings.npy", protbert_embeddings)
print("file protbert_embeddings.npy saved")

# ======== Blomap Embeddings ========
def add_padding(sequence, max_length):
    return sequence.ljust(max_length, 'X')

def blomap_extra_encode(sequence):
    blomap_dict = {
        'A': [0.62, 0.29, 0.00, -0.06, 0.00], 'R': [-2.53, 0.00, 3.00, 1.17, 0.00],
        'N': [-0.78, 0.58, 0.20, 0.00, 0.00], 'D': [-0.90, 0.46, 1.60, 0.00, 0.00],
        'C': [0.29, -0.10, 0.00, 0.00, 0.00], 'Q': [-0.85, 0.00, 0.20, 0.00, 0.00],
        'E': [-0.74, 0.00, 1.40, 0.00, 0.00], 'G': [0.48, 0.00, 0.00, 0.00, 0.00],
        'H': [-0.40, 0.00, 1.00, 0.00, 0.00], 'I': [1.38, -0.20, 0.00, 0.00, 0.00],
        'L': [1.06, -0.20, 0.00, 0.00, 0.00], 'K': [-1.50, 0.00, 2.90, 0.00, 0.00],
        'M': [0.64, -0.10, 0.00, 0.00, 0.00], 'F': [1.19, -0.30, 0.00, 0.00, 0.00],
        'P': [0.12, 0.00, 0.00, 0.00, 0.00], 'S': [-0.18, 0.00, 0.00, 0.00, 0.00],
        'T': [-0.05, 0.00, 0.00, 0.00, 0.00], 'W': [0.81, -0.30, 0.00, 0.00, 0.00],
        'Y': [0.26, -0.20, 0.00, 0.00, 0.00], 'V': [1.08, -0.10, 0.00, 0.00, 0.00],
        'X': [0.00, 0.00, 0.00, 0.00, 0.00]
    }
    encoding = [blomap_dict.get(aa, [0, 0, 0, 0, 0]) for aa in sequence]
    return np.array(encoding).flatten()

SEQ_LEN = df["standard_sequence"].dropna().apply(len).max()
df["adjusted_sequence"] = df["standard_sequence"].apply(lambda s: add_padding(s, SEQ_LEN) if isinstance(s, str) else s)
blomap_embeddings = np.array([blomap_extra_encode(seq) for seq in df["adjusted_sequence"].dropna()])
np.save("blomap_embeddings.npy", blomap_embeddings)
print("file blomap_embeddings.npy saved")

# ======== Morgan Fingerprints ========
def calculate_fingerprints(smiles):
    if not isinstance(smiles, str):
        return np.zeros(2048, dtype=int)
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        generator = GetMorganGenerator(radius=2, fpSize=2048)
        fp = generator.GetFingerprint(mol)
        return np.array(fp, dtype=int)
    return np.zeros(2048, dtype=int)

if "smiles_sequence" in df.columns:
    df["smiles_sequence"] = df["smiles_sequence"].fillna("")
    fingerprints = np.vstack(df["smiles_sequence"].apply(calculate_fingerprints))
    np.save("fingerprints.npy", fingerprints)
    print("file fingerprints.npy saved")
