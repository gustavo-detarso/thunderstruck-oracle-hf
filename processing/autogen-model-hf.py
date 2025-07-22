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

