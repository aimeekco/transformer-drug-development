from transformers import AutoTokenizer, AutoModel 

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# tokenize SMILES strings 
input = tokenizer(data['smiles'].tolist(), return_tensors='pt', padding=True, truncation=True)

#generate embeddings
with torch.no_grad():
    output = model(**input)
    
embeddings = outputs.last_hidden_state