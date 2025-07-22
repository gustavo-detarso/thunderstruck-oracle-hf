import os
import glob
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

qa_folder = "data_qa"
faiss_index_path = "faiss_chunks.index"
mapping_path = "faiss_mapping.pkl"
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

qas = []
origins = []
seen = set()
for qa_path in glob.glob(os.path.join(qa_folder, "**", "*_qa.jsonl"), recursive=True):
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            qa = json.loads(line)
            contexto = qa["pergunta"] + " " + qa["resposta"]
            if contexto not in seen:
                seen.add(contexto)
                qas.append(contexto)
                origins.append({
                    "arquivo": os.path.basename(qa_path),
                    "pergunta": qa["pergunta"],
                    "resposta": qa["resposta"]
                })

if not qas:
    print("Nenhum chunk para indexar! Cheque seus arquivos em data_qa.")
    exit(1)

print(f"Total de chunks QA para indexação: {len(qas)}")

model = SentenceTransformer(embedding_model)
embeddings = model.encode(qas, show_progress_bar=True, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
faiss.write_index(index, faiss_index_path)

with open(mapping_path, "wb") as f:
    pickle.dump(origins, f)

print(f"Indexação completa! Index salvo em {faiss_index_path} e mapping em {mapping_path}")

