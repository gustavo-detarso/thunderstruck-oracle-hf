import os
import glob
import faiss
import pickle
import json
import datetime
import re
import pandas as pd
import unicodedata
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel

faiss_index_path = "faiss_chunks.index"
mapping_path = "faiss_mapping.pkl"
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
base_model_id = "meta-llama/Meta-Llama-3-8B"
lora_dir = "./lora_llama_finetuned"

if os.path.exists(os.path.join(lora_dir, "tokenizer_config.json")):
    tokenizer_path = lora_dir
else:
    tokenizer_path = base_model_id
tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = LlamaForCausalLM.from_pretrained(base_model_id)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, lora_dir)
model.eval()

index = faiss.read_index(faiss_index_path)
with open(mapping_path, "rb") as f:
    origins = pickle.load(f)
embedder = SentenceTransformer(embedding_model)

USER_QA_LOG = os.path.join("data_qa", "user_questions.jsonl")
USER_QA_TRAIN = os.path.join("data_qa", "user_questions_qa.jsonl")
USER_QA_INVALID = os.path.join("data_qa", "user_questions_invalid.jsonl")

def normalizar_str(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in s if not unicodedata.combining(c)])
    s = re.sub(r'[^a-z0-9 ]', '', s)  # remove tudo que não é letra, número ou espaço
    return s

def extract_table_from_txt(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("#TABELA:"):
            table_lines = []
            for l in lines[i+1:]:
                l_strip = l.strip()
                if not l_strip:
                    break
                table_lines.append(l_strip)
            if len(table_lines) < 2:
                continue
            cols = [c.strip().lower() for c in table_lines[0].split(",")]
            rows = []
            for r in table_lines[1:]:
                vals = [v.strip() for v in r.split(",")]
                if len(vals) == len(cols):
                    rows.append(vals)
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=cols)
            return df
    return None

def extract_estado(pergunta):
    estados = [
        ("acre","ac"),("alagoas","al"),("amapa","ap"),("amazonas","am"),("bahia","ba"),("ceara","ce"),
        ("distrito federal","df"),("espirito santo","es"),("goias","go"),("maranhao","ma"),("mato grosso","mt"),
        ("mato grosso do sul","ms"),("minas gerais","mg"),("para","pa"),("paraiba","pb"),("parana","pr"),
        ("pernambuco","pe"),("piaui","pi"),("rio de janeiro","rj"),("rio grande do norte","rn"),
        ("rio grande do sul","rs"),("rondonia","ro"),("roraima","rr"),("santa catarina","sc"),
        ("sao paulo","sp"),("sergipe","se"),("tocantins","to")
    ]
    p = normalizar_str(pergunta)
    for nome, sigla in estados:
        if nome in p or f" {sigla} " in f" {p} ":
            return nome
    return None

def is_table_question(pergunta):
    keywords = ["quantos", "quantidade", "total", "número de", "contar", "contagem"]
    return any(kw in pergunta.lower() for kw in keywords)

def is_listar_nomes_question(pergunta):
    keywords = [
        "quais", "quem", "nomes", "lista", "convocados", "aprovados",
        "todos os nomes", "nomes dos convocados", "listar nomes", "nomes aprovados", "listagem de nomes"
    ]
    p = pergunta.lower()
    # Também detecta frases tipo "todos os nomes" mesmo em perguntas longas
    if any(kw in p for kw in keywords) and extract_estado(pergunta) is not None:
        return True
    # Busca por padrões tipo 'todos os nomes', 'nome(s)'... para cobrir variações
    if re.search(r'\bnome[s]?\b', p) and extract_estado(pergunta) is not None:
        return True
    return False

def search_table_for_estado(pergunta):
    estado = extract_estado(pergunta)
    if not estado:
        return None, "Não encontrei estado na pergunta."
    for txt_path in glob.glob("data/**/*.txt", recursive=True):
        df = extract_table_from_txt(txt_path)
        if df is not None and "estado" in df.columns:
            df["estado_norm"] = df["estado"].apply(normalizar_str)
            estado_norm = normalizar_str(estado)
            match = df[df["estado_norm"] == estado_norm]
            if not match.empty:
                if "convocados" in df.columns:
                    try:
                        qtd = match["convocados"].astype(int).sum()
                    except Exception:
                        qtd = len(match)
                else:
                    qtd = len(match)
                return str(qtd), f"[Tabela extraída de {os.path.basename(txt_path)}]"
    return None, f"Não encontrei o estado '{estado}' nas tabelas dos arquivos .txt."

def search_table_for_nomes_estado(pergunta):
    estado = extract_estado(pergunta)
    if not estado:
        return None, "Não encontrei estado na pergunta."
    estado_norm = normalizar_str(estado)
    encontrou = False
    for txt_path in glob.glob("data/**/*.txt", recursive=True):
        df = extract_table_from_txt(txt_path)
        if df is not None and "estado" in df.columns:
            df["estado_norm"] = df["estado"].apply(normalizar_str)
            print(f"[DEBUG] Arquivo: {txt_path}")
            print("[DEBUG] Valores únicos em estado_norm:", df["estado_norm"].unique())
            print("[DEBUG] Estado buscado:", estado_norm)
            match = df[df["estado_norm"] == estado_norm]
            print(f"Entradas filtradas para '{estado_norm}':", len(match))
            if not match.empty:
                encontrou = True
                for nome_col in ["nome", "candidato", "participante"]:
                    if nome_col in df.columns:
                        nomes = match[nome_col].tolist()
                        nomes_limpos = [n for n in nomes if n and n != "-"]
                        if nomes_limpos:
                            print("[DEBUG] Nomes extraídos:", nomes_limpos[:5], "...")
                            return "\n".join(nomes_limpos), f"[Nomes extraídos de {os.path.basename(txt_path)}]"
    if not encontrou:
        print(f"[DEBUG] Estado '{estado_norm}' não encontrado em nenhuma tabela txt.")
    return None, f"Não encontrei os nomes do estado '{estado}' nas tabelas dos arquivos .txt."

def resposta_literal(question):
    for c in origins:
        if c["pergunta"].strip().lower() == question.strip().lower():
            return c["resposta"]
    return None

def ask_rag(question, top_k=2, max_new_tokens=100):
    # 1. Se for pergunta de listar nomes (ex: "Quais são os convocados do Maranhão?"), prioriza busca tabular!
    if is_listar_nomes_question(question):
        resposta, fonte = search_table_for_nomes_estado(question)
        if resposta:
            return resposta, [fonte]
        else:
            return fonte, [fonte]

    # 2. Busca resposta literal direta (QA exato)
    resposta_lit = resposta_literal(question)
    if resposta_lit is not None:
        return resposta_lit, ["[Resposta literal do dataset de QA]"]

    # 3. Consulta tabular para quantidade/contagem (ex: "Quantos convocados do Maranhão?")
    if is_table_question(question):
        resposta, fonte = search_table_for_estado(question)
        if resposta:
            return resposta, [fonte]
        else:
            return fonte, [fonte]

    # 4. Fallback: RAG+LLM normal
    emb = embedder.encode([question], normalize_embeddings=True)
    D, I = index.search(emb, top_k)
    contextos = [origins[i] for i in I[0]]
    contexto_texto = "\n".join([f"Trecho {j+1}: {c['pergunta']} {c['resposta']}" for j, c in enumerate(contextos)])
    prompt = f"""Baseie-se apenas nos dados abaixo para responder de forma literal à pergunta.

Contexto:
{contexto_texto}

Pergunta: {question}
Responda usando exatamente o texto da resposta presente no contexto. 
Se não encontrar, diga 'Não encontrado no contexto'.
Resposta:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
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
        print("\nResposta do modelo:\n", resposta)
        print("\nContexto(s) utilizado(s):")
        for c in contextos:
            print(f"- {c}")
        avaliacao = input("\nEssa resposta te ajudou? [s/n]: ").strip().lower()
        while avaliacao not in ("s", "n"):
            avaliacao = input("Por favor, digite 's' para sim ou 'n' para não: ").strip().lower()
        log_user_qa(question, resposta, contextos, avaliacao)
        if avaliacao == "s":
            print("Pergunta/resposta adicionada ao dataset de treino (data_qa/user_questions_qa.jsonl).")
        else:
            print("Pergunta/resposta salva para revisão em data_qa/user_questions_invalid.jsonl.")

