import faiss
import pickle
import json
import os
import datetime
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel

faiss_index_path = "faiss_chunks.index"
mapping_path = "faiss_mapping.pkl"
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
base_model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
lora_dir = "./lora_llama_finetuned"
USER_QA_LOG = os.path.join("data_qa", "user_questions.jsonl")
USER_QA_TRAIN = os.path.join("data_qa", "user_questions_qa.jsonl")
USER_QA_INVALID = os.path.join("data_qa", "user_questions_invalid.jsonl")  # <-- atualizado

index = faiss.read_index(faiss_index_path)
with open(mapping_path, "rb") as f:
    origins = pickle.load(f)

embedder = SentenceTransformer(embedding_model)
tokenizer = LlamaTokenizerFast.from_pretrained(base_model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
base_model = LlamaForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, lora_dir)

def ask_rag(question, top_k=2, max_new_tokens=100):
    emb = embedder.encode([question], normalize_embeddings=True)
    D, I = index.search(emb, top_k)
    contextos = [origins[i] for i in I[0]]
    contexto_texto = "\n".join([f"Trecho {j+1}: {c['pergunta']} {c['resposta']}" for j, c in enumerate(contextos)])
    prompt = f"""Contexto:\n{contexto_texto}\n\nPergunta: {question}\nResposta:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    resposta = tokenizer.decode(output[0], skip_special_tokens=True)
    resposta_limpa = resposta.split("Resposta:")[-1].strip()
    return resposta_limpa, contextos

def log_user_qa(pergunta, resposta, contextos, avaliacao):
    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pergunta": pergunta,
        "resposta": resposta,
        "contextos": contextos,
        "avaliacao": avaliacao
    }
    with open(USER_QA_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(USER_QA_TRAIN), exist_ok=True)
    qa_line = {
        "pergunta": pergunta,
        "resposta": resposta,
        "validada": avaliacao.lower() == "s"
    }
    if avaliacao.lower() == "s":
        with open(USER_QA_TRAIN, "a", encoding="utf-8") as f:
            f.write(json.dumps(qa_line, ensure_ascii=False) + "\n")
    else:
        with open(USER_QA_INVALID, "a", encoding="utf-8") as f:
            f.write(json.dumps(qa_line, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    print("RAG QA - Pergunte sobre seus documentos!")
    while True:
        question = input("\nDigite sua pergunta (ou 'sair'): ").strip()
        if question.lower() in ["sair", "exit", "quit"]:
            break
        resposta, contextos = ask_rag(question)
        print("\nResposta do modelo:", resposta)
        print("\nContexto(s) utilizado(s):")
        for c in contextos:
            print(f"- [{c['arquivo']}] Q: {c['pergunta']} | A: {c['resposta']}")
        # Avaliação do usuário
        avaliacao = input("\nEssa resposta te ajudou? [s/n]: ").strip().lower()
        while avaliacao not in ("s", "n"):
            avaliacao = input("Por favor, digite 's' para sim ou 'n' para não: ").strip().lower()
        # Salva logs e só adiciona ao treino se foi validada
        log_user_qa(question, resposta, contextos, avaliacao)
        if avaliacao == "s":
            print("Pergunta/resposta adicionada ao dataset de treino (data_qa/user_questions_qa.jsonl).")
        else:
            print("Pergunta/resposta salva para revisão em data_qa/user_questions_invalid.jsonl.")

