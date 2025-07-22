#!/usr/bin/env bash
set -euo pipefail

# 0) Defina o nome do diretório raiz
BASE_DIR="thunderstruck-oracle-hf"

# 1) Cria o diretório raiz e entra nele
mkdir -p "${BASE_DIR}"
cd "${BASE_DIR}"

# 2) Cria a estrutura de pastas internas
mkdir -p data data_qa preprocessing processing

# 3) Arquivos na raiz

## .env
cat << 'EOF' > .env
OPENAI_API_KEY='seu_token'
HUGGINGFACE_TOKEN=seu_token
EOF

## requirements.txt
cat << 'EOF' > requirements.txt
# Deep Learning Core
torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
transformers>=4.41.2,<4.44
peft>=0.10.0
bitsandbytes>=0.43.0

# Datasets e aceleração
datasets>=2.19
accelerate>=0.29

# Embeddings, utilitários
sentencepiece
faiss-cpu
sentence-transformers

# Data Science e métricas
pandas
tqdm
rich
scikit-learn>=1.4
scipy
numpy>=1.26,<2.2.0

# Outros
openai
huggingface_hub
python-dotenv
requests

# TensorFlow (opcional, se for usar tf-keras)
tf-keras
EOF

## rag_qa_finetuned.py
cat << 'EOF' > rag_qa_finetuned.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera arquivo de perguntas
"""
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
EOF

# 4) Scripts de pré-processamento
cat << 'EOF' > preprocessing/gen_qa_chatgpt_from_txt.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera dataset de QA usando o ChatGPT a partir de arquivos .txt
"""
import os
import re
import json
import glob
import hashlib
import openai
import time
from dotenv import load_dotenv
from tqdm import tqdm
from itertools import combinations

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def find_column(cols, aliases):
    aliases_lower = {a.lower() for a in aliases}
    for c in cols:
        if c.lower() in aliases_lower:
            return c
    for c in cols:
        for a in aliases_lower:
            if c.lower().startswith(a):
                return c
    return None

def call_gpt_api(prompt, model="gpt-4o", retries=3):
    for attempt in range(1, retries+1):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=4096,
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            qas = data.get("qas", data if isinstance(data, list) else [])
            if not isinstance(qas, list):
                raise ValueError("Campo 'qas' não é uma lista.")
            qas = [qa for qa in qas if isinstance(qa, dict) and 'pergunta' in qa and 'resposta' in qa]
            return qas
        except Exception as e:
            print(f"[API parse erro {attempt}/{retries}]: {e}")
            time.sleep(1)
    print("⚠️ Falha ao obter JSON válido, ignorando este bloco.")
    return []

def extrair_tags(texto):
    matches = re.findall(r"#TAG_([A-Z_]+):\s*(.*)", texto, re.IGNORECASE)
    return {k.lower(): v.strip() for k, v in matches}

def extrair_tabela(texto):
    m = re.search(r"#TABELA:.*?\n(.*)", texto, re.DOTALL)
    if not m:
        return None, None
    linhas = [l for l in m.group(1).splitlines() if l.strip()]
    cols = [c.strip() for c in linhas[0].split(',')]
    rows = []
    for l in linhas[1:]:
        vals = [v.strip() for v in l.split(',')]
        if len(vals) == len(cols):
            rows.append(vals)
    return cols, rows

def dedup(qas):
    seen = set()
    unique = []
    for qa in qas:
        if not isinstance(qa, dict):
            continue
        pergunta = str(qa.get('pergunta','')).strip()
        resposta = str(qa.get('resposta','')).strip()
        if not pergunta or not resposta:
            continue
        key = pergunta + '|' + resposta
        h = hashlib.sha256(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append({"pergunta": pergunta, "resposta": resposta})
    return unique

def prompt_negativas(cols, rows):
    preview = "\n".join([",".join(cols)] + [",".join(r) for r in rows[:15]] + (["..."] if len(rows)>15 else []))
    exemplos = [f"Existe 'XYZ' em '{c}'?" for c in cols] + [f"Registro nº1000 em '{c}'?" for c in cols]
    ex_str = "\n- ".join(exemplos)
    return f"""Apenas gere o seguinte JSON e nada além disso:
{{
  "qas": []
}}

Gere perguntas negativas e respectivas respostas sobre a tabela a seguir:
- {ex_str}

Tabela:
{preview}
"""

def prompt_sinonimos(cols, rows):
    preview = "\n".join([",".join(cols)] + [",".join(r) for r in rows[:15]] + (["..."] if len(rows)>15 else []))
    exemplos = [f"Liste variações de '{c}'." for c in cols][:5]
    ex_str = "\n- ".join(exemplos)
    return f"""Apenas gere o seguinte JSON e nada além disso:
{{
  "qas": []
}}

Gere perguntas e respostas envolvendo sinônimos e variações de termos das colunas da tabela:
- {ex_str}

Tabela:
{preview}
"""

def prompt_combinadas(cols, rows):
    preview = "\n".join([",".join(cols)] + [",".join(r) for r in rows[:15]] + (["..."] if len(rows)>15 else []))
    combo2 = [f"Qual '{b}' quando '{a}'=X?" for a,b in combinations(cols,2)][:3]
    combo3 = [f"Qual '{c}' para '{a}'=X e '{b}'=Y?" for a,b,c in combinations(cols,3)][:2]
    ex_str = "\n- ".join(combo2 + combo3)
    return f"""Apenas gere o seguinte JSON e nada além disso:
{{
  "qas": []
}}

Gere perguntas e respostas combinando 2 ou mais colunas da tabela:
- {ex_str}

Tabela:
{preview}
"""

def gerar_programatico(cols, rows):
    dados = [dict(zip(cols, r)) for r in rows]
    out = []
    # Valores únicos e repetidos
    for c in cols:
        vals = sorted({d.get(c,'') for d in dados if d.get(c)})
        out.append({
            'pergunta': f"Quais valores únicos da coluna '{c}'?",
            'resposta': "; ".join(vals) if vals else "Nenhum valor encontrado."
        })
        reps = [v for v in vals if sum(1 for d in dados if d.get(c)==v) > 1]
        if reps:
            out.append({
                'pergunta': f"Há valores repetidos em '{c}'?",
                'resposta': ", ".join(reps)
            })
    # Contagens por coluna
    for c in cols:
        vals = [d.get(c,'') for d in dados if d.get(c)]
        out.append({
            'pergunta': f"Quantos registros distintos há para a coluna '{c}'?",
            'resposta': str(len(set(vals)))
        })
    # Todas as relações cruzadas (1:1) entre as 3 primeiras colunas
    if len(cols) >= 3:
        c1, c2, c3 = cols[:3]
        for d in dados:
            v1, v2, v3 = d[c1], d[c2], d[c3]
            if v1 and v2:
                out.append({"pergunta": f"Qual {c2} está relacionado a {c1} '{v1}'?", "resposta": v2})
                out.append({"pergunta": f"Qual {c1} corresponde a {c2} '{v2}'?", "resposta": v1})
            if v1 and v3:
                out.append({"pergunta": f"Qual {c3} está relacionado a {c1} '{v1}'?", "resposta": v3})
                out.append({"pergunta": f"Qual {c1} corresponde a {c3} '{v3}'?", "resposta": v1})
            if v2 and v3:
                out.append({"pergunta": f"Qual {c3} está relacionado a {c2} '{v2}'?", "resposta": v3})
                out.append({"pergunta": f"Qual {c2} corresponde a {c3} '{v3}'?", "resposta": v2})
    return out

def detectar_tipo_documento(texto):
    tags = extrair_tags(texto)
    tipo = tags.get('tipo_documento', '').lower()
    if 'edital' in tipo:
        return 'edital'
    if 'portaria' in tipo:
        return 'portaria'
    return 'outro'

def gerar_qas_generico(texto, tipo="outro"):
    qas = []
    tags = extrair_tags(texto)
    if tags:
        campos = "\n".join(f"{k}: {v}" for k,v in tags.items())
        try:
            qas += call_gpt_api(
                f"""Apenas gere o seguinte JSON e nada além disso:
{{
  "qas": []
}}

Gere perguntas e respostas relevantes baseadas nos campos abaixo, sem repetir perguntas que possam ser feitas a partir da tabela:
{campos}
"""
            )
        except:
            pass
    cols, rows = extrair_tabela(texto)
    if cols and rows:
        qas += gerar_programatico(cols, rows)
        for fn in (prompt_negativas, prompt_sinonimos, prompt_combinadas):
            try:
                qas += call_gpt_api(fn(cols, rows))
            except:
                pass
    return dedup(qas)

def gerar_qas_edital(texto):
    return gerar_qas_generico(texto, tipo="edital")

def gerar_qas_portaria(texto):
    return gerar_qas_generico(texto, tipo="portaria")

def gerar_qas(txt_path):
    texto = open(txt_path, encoding='utf-8').read()
    tipo = detectar_tipo_documento(texto)
    if tipo == 'edital':
        return gerar_qas_edital(texto)
    elif tipo == 'portaria':
        return gerar_qas_portaria(texto)
    else:
        return gerar_qas_generico(texto)

def main():
    os.makedirs("data_qa", exist_ok=True)
    for txt in tqdm(glob.glob("data/**/*.txt", recursive=True), desc="Processando arquivos TXT"):
        try:
            qas = gerar_qas(txt)
            if not qas:
                print(f"Nenhuma QA para {txt}")
                continue
            out_path = f"data_qa/{os.path.basename(txt).replace('.txt','_qa.jsonl')}"
            with open(out_path, "w", encoding="utf-8") as fo:
                for qa in qas:
                    fo.write(json.dumps(qa, ensure_ascii=False)+"\n")
            print(f"Gerado: {out_path} ({len(qas)} perguntas e respostas salvas)")
        except Exception as err:
            print(f"[FALHA]: {txt} Erro:{err}")

if __name__ == "__main__":
    main()
EOF
chmod +x preprocessing/*.py

# 5) Scripts de processamento
cat << 'EOF' > processing/autogen-model-hf.py
#!/usr/bin/env python3
"""
Autogenerates QA prompts using HuggingFace model.
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # para rastrear erros CUDA corretamente
import glob
import json
import math
import random
import torch
import numpy as np
from datetime import datetime
import time
import requests
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizerFast, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# 0) Pergunta o valor do droplet por hora
while True:
    try:
        droplet_per_hour = float(input("Digite o valor do seu droplet (US$/hora): ").replace(",", "."))
        break
    except Exception:
        print("Valor inválido, tente novamente. Exemplo: 0.7")

inicio_treino = time.time()

# 1) Seeds e dispositivo
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Dataset de QA
def build_qa_dataset(data_dir="data_qa"):
    paths = glob.glob(os.path.join(data_dir, "**", "*_qa.jsonl"), recursive=True)
    examples = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    j = json.loads(ln)
                    q, a = j["pergunta"], j["resposta"]
                    examples.append({"text": f"Pergunta: {q}\nResposta: {a}"})
    return examples

pairs        = build_qa_dataset()
for p in pairs[:5]:
    print(repr(p["text"]))

train_p, val_p = train_test_split(pairs, test_size=0.05, random_state=SEED)
train_ds     = Dataset.from_list(train_p)
val_ds       = Dataset.from_list(val_p)

# 3) Tokenizer e tokenização
MODEL = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizerFast.from_pretrained(MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
val_ds   = val_ds.map(tokenize, batched=True, remove_columns=["text"])

# 4) Collate_fn para DataLoader
def collate_fn(batch):
    input_ids      = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
    labels         = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, collate_fn=collate_fn)

# 5) Calcula steps por epoch e número total de epochs
grad_accumulation = 8
steps_per_epoch = math.ceil(len(train_loader) / grad_accumulation)
total_steps     = 1000
max_epochs      = math.ceil(total_steps / steps_per_epoch)

# 6) Carrega modelo e aplica LoRA
base_model = LlamaForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,  # FP32 para evitar overflow de NaN
)
base_model.resize_token_embeddings(len(tokenizer))

peft_conf = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)
model = get_peft_model(base_model, peft_conf)
model.config.use_cache = False
model.to(DEVICE).train()

# 7) Otimizador, scheduler e TensorBoard
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-6,
)
warmup_steps = int(0.1 * total_steps)
scheduler    = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
writer = SummaryWriter(log_dir=f"./runs/{datetime.now():%Y%m%d_%H%M%S}")

# 8) Loop de treinamento manual com barra de progresso de Epochs e batches
step = 0
optimizer.zero_grad()

print(f"=== Iniciando treino: {max_epochs} epochs, {total_steps} steps total ===")
for epoch in range(1, max_epochs + 1):
    epoch_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
    for batch_idx, batch in enumerate(epoch_loop, start=1):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / grad_accumulation
        loss.backward()

        if batch_idx % grad_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            writer.add_scalar("train/loss", loss.item() * grad_accumulation, step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

            if step % 50 == 0:
                print(f"[Step {step}/{total_steps}] loss: {(loss.item() * grad_accumulation):.4f}")

            if step >= total_steps:
                break
    if step >= total_steps:
        break

# 9) Avaliação: perplexity no conjunto de validação
model.eval()
total_val_loss = 0.0
count = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_val_loss += outputs.loss.item()
        count += 1
avg_val_loss = total_val_loss / count
perplexity = math.exp(avg_val_loss)
print(f"Validation Perplexity: {perplexity:.2f}")
writer.add_scalar("eval/perplexity", perplexity)

# 10) Inferência exemplo
model.eval()
def generate_response(prompt: str, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("Exemplo de geração:")
print(generate_response("Pergunta: Qual é a capital do Brasil?\nResposta:"))

# 10.1) Salvar o adapter LoRA para uso futuro (inference incremental com PeftModel)
lora_adapter_dir = "./lora_llama_finetuned"
model.save_pretrained(lora_adapter_dir)
print(f"Adapter LoRA salvo em {lora_adapter_dir}")

# 11) Mescla adapters LoRA
print("Mesclando adapters LoRA no modelo base...")
merged_model = model.merge_and_unload()
merged_dir = "./merged_model"
merged_model.save_pretrained(merged_dir)
print(f"Modelo mesclado salvo em {merged_dir}")

# 12) Push ao HF Hub
REPO_ID = "seu-username/seu-repo"
merged_model.push_to_hub(REPO_ID, use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
tokenizer.push_to_hub(REPO_ID, use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
print(f"Modelo e tokenizer enviados para o Hub: {REPO_ID}")

# 13) Cálculo de tempo total, custo em dólares e reais
fim_treino = time.time()
tempo_total_s = fim_treino - inicio_treino
tempo_total_h = tempo_total_s / 3600.0
custo_total_usd = tempo_total_h * droplet_per_hour

# Cotação dólar turismo (AwesomeAPI)
try:
    r = requests.get("https://economia.awesomeapi.com.br/json/last/USD-BRL")
    dolar_brl = float(r.json()["USDBRL"]["high"])
    custo_total_brl = custo_total_usd * dolar_brl
    print(f"\n=== Estatísticas de custo ===")
    print(f"Tempo total de execução: {tempo_total_h:.2f} horas")
    print(f"Custo estimado do droplet: US$ {custo_total_usd:.2f}")
    print(f"Cotação dólar turismo: R$ {dolar_brl:.2f}")
    print(f"Custo estimado em reais: R$ {custo_total_brl:.2f}")
except Exception as e:
    print("Falha ao consultar a cotação do dólar. Erro:", e)
    print(f"Tempo total: {tempo_total_h:.2f}h, custo US$: {custo_total_usd:.2f}")
EOF

cat << 'EOF' > processing/index_chunks_faiss.py
#!/usr/bin/env python3
"""
Autogenerates index FAISS.
"""
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
EOF
chmod +x processing/*.py

echo "✅ Projeto '${BASE_DIR}' inicializado com sucesso!"

