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

