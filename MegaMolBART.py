from transformers import BartForConditionalGeneration, BartTokenizer

# clobetasol proprionate SMILES: CC(=O)OC1C(C(C2C1C3CCC4=CC(=O)CCC4(C3(C2(C(=O)CO)C)O)C)Cl)OC(=O)C

model_name = "seyonec/PubChem10M_SMILES_BART"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# SMILES string 
smiles = "CC(=O)OC1C(C(C2C1C3CCC4=CC(=O)CCC4(C3(C2(C(=O)CO)C)O)C)Cl)OC(=O)C"
inputs = tokenizer(smiles, return_tensors="pt")

#generate docking patterns
outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=5)
docking_patterns = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

