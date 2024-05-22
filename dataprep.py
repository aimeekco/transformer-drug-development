from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

data = pd.read_csv('drug_target_data.csv')

# convert SMILES to RDKit mol object
data['rdkit_mol'] = data['smiles'].apply(Chem.MolFromSmiles)

# generate molecular fingerprints 
data['fingerprints'] = data['rdkit_mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))