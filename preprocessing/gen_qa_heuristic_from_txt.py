import os
import re
import json
import glob
import hashlib
from collections import Counter, defaultdict

def hash_arquivo(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def carregar_hashes(path="data_qa/hashes.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_hashes(hashes, path="data_qa/hashes.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)

def gerar_variacoes_tabela(col, valor, resposta, contexto=None):
    variacoes = [
        f"Qual o valor da coluna '{col}' igual a '{valor}'?",
        f"Me informe o {col} cujo valor é '{valor}'.",
        f"Existe algum registro com '{col}' = '{valor}'?",
        f"Liste todos os registros onde '{col}' é '{valor}'.",
        f"Há algum registro em que '{col}' corresponda a '{valor}'?",
    ]
    if contexto:
        variacoes.append(f"No contexto de {contexto}, qual é o valor de '{col}' igual a '{valor}'?")
    return [{"pergunta": p, "resposta": resposta} for p in variacoes]

def gerar_variacoes_par_col(col1, val1, col2, val2):
    return [
        {"pergunta": f"Quando '{col1}' é '{val1}', qual o valor de '{col2}'?", "resposta": val2},
        {"pergunta": f"Se '{col1}' for '{val1}', informe o '{col2}'.", "resposta": val2},
        {"pergunta": f"O valor de '{col2}' correspondente a '{col1}' igual a '{val1}'?", "resposta": val2}
    ]

def gerar_variacoes_texto(bloco, arquivo_nome, titulo="Trecho"):
    bloco_limpo = bloco.strip().replace('\n', ' ')
    if len(bloco_limpo) < 20:
        return []
    variacoes = [
        f"O que diz o seguinte trecho do documento '{arquivo_nome}': \"{bloco_limpo[:80]}...\"?",
        f"Explique o seguinte trecho do documento '{arquivo_nome}': \"{bloco_limpo[:80]}...\"",
        f"Resuma o trecho: \"{bloco_limpo[:80]}...\"",
        f"Qual a informação presente neste trecho: \"{bloco_limpo[:80]}...\"?",
        f"O que se pode concluir a partir do seguinte texto do documento '{arquivo_nome}'?",
    ]
    return [{"pergunta": p, "resposta": bloco_limpo} for p in variacoes]

def gerar_qas_tabela(cabecalho, linhas, arquivo_nome, documento_nome):
    qas = []
    dados = [dict(zip(cabecalho, l)) for l in linhas if len(l) == len(cabecalho)]
    N = len(dados)

    for i, row in enumerate(dados):
        for col in cabecalho:
            valor = row[col]
            qas.extend(gerar_variacoes_tabela(col, valor, valor))
        for c1 in cabecalho:
            for c2 in cabecalho:
                if c1 != c2:
                    qas.extend(gerar_variacoes_par_col(c1, row[c1], c2, row[c2]))

    for col in cabecalho:
        unicos = list({row[col] for row in dados})
        qas.append({
            "pergunta": f"Quais são todos os valores únicos da coluna '{col}' no arquivo {arquivo_nome}?",
            "resposta": "; ".join(unicos)
        })
        qas.append({
            "pergunta": f"Quantos valores diferentes existem na coluna '{col}'?",
            "resposta": str(len(unicos))
        })
        counter = Counter([row[col] for row in dados])
        mais_comum, freq = counter.most_common(1)[0]
        qas.append({
            "pergunta": f"Qual o valor mais comum da coluna '{col}'?",
            "resposta": f"{mais_comum} ({freq} ocorrências)"
        })
        top3 = counter.most_common(3)
        qas.append({
            "pergunta": f"Quais os três valores mais frequentes da coluna '{col}'?",
            "resposta": "; ".join([f"{v} ({f})" for v, f in top3])
        })
        iniciais = set([row[col][0].upper() for row in dados if row[col]])
        for letra in iniciais:
            lista = [row[col] for row in dados if row[col].upper().startswith(letra)]
            if lista:
                qas.append({
                    "pergunta": f"Quantos registros da coluna '{col}' começam com a letra '{letra}'?",
                    "resposta": f"{len(lista)}"
                })
                qas.append({
                    "pergunta": f"Liste todos os valores da coluna '{col}' que começam com a letra '{letra}'.",
                    "resposta": "; ".join(lista)
                })
        if col.lower() in ["nome", "nome completo", "unidade", "municipio"]:
            for substr in ["SILVA", "MARIA", "SANTOS"]:
                lista = [row[col] for row in dados if substr in row[col].upper()]
                if lista:
                    qas.append({
                        "pergunta": f"Quantos registros da coluna '{col}' contêm '{substr.title()}'?",
                        "resposta": str(len(lista))
                    })
                    qas.append({
                        "pergunta": f"Liste todos os valores da coluna '{col}' que contêm '{substr.title()}'.",
                        "resposta": "; ".join(lista)
                    })

    for c1 in cabecalho:
        for c2 in cabecalho:
            if c1 != c2:
                agrupamento = defaultdict(list)
                for row in dados:
                    agrupamento[row[c1]].append(row[c2])
                for v1, lista in agrupamento.items():
                    qas.append({
                        "pergunta": f"Quais valores da coluna '{c2}' existem para '{c1}' = '{v1}'?",
                        "resposta": "; ".join(sorted(set(lista)))
                    })
                    qas.append({
                        "pergunta": f"Quantos valores da coluna '{c2}' existem para '{c1}' = '{v1}'?",
                        "resposta": str(len(set(lista)))
                    })

    qas.append({
        "pergunta": f"Quantos registros existem na tabela do arquivo {arquivo_nome}?",
        "resposta": str(N)
    })
    qas.append({
        "pergunta": f"Qual é o primeiro registro da tabela do arquivo {arquivo_nome}?",
        "resposta": "; ".join([row[col] for col in cabecalho for row in [dados[0]]])
    })
    qas.append({
        "pergunta": f"Qual é o último registro da tabela do arquivo {arquivo_nome}?",
        "resposta": "; ".join([row[col] for col in cabecalho for row in [dados[-1]]])
    })
    return qas

def processar_arquivo_txt(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        texto = f.read()
    fname = os.path.basename(txt_path)
    qas = []

    # QAs para texto corrido via TAGs
    ementa_match = re.search(r"#TAG_EMENTA:([^\n]*)", texto)
    if ementa_match:
        qas.extend(gerar_variacoes_texto(ementa_match.group(1), fname, "Ementa"))
    corpo_match = re.search(r"#TAG_CORPO:\n(.*?)(?=#TAG_|#TABELA:|$)", texto, re.DOTALL)
    if corpo_match:
        qas.extend(gerar_variacoes_texto(corpo_match.group(1), fname, "Corpo"))
    assinatura_match = re.search(r"#TAG_ASSINATURA:([^\n]*)", texto)
    if assinatura_match:
        qas.extend(gerar_variacoes_texto(assinatura_match.group(1), fname, "Assinatura"))
    # Artigos
    for m in re.finditer(r"(Art\.? ?\d+\.? ?[^\n]+)", texto):
        qas.extend(gerar_variacoes_texto(m.group(1), fname, "Artigo"))

    # QAs para tabelas
    tabela_match = re.search(r"#TABELA:.*?\n(.*)", texto, re.DOTALL)
    if tabela_match:
        linhas_brutas = [l for l in tabela_match.group(1).strip().split('\n') if l.strip()]
        if linhas_brutas:
            cabecalho = [x.strip() for x in linhas_brutas[0].split(',')]
            linhas = []
            for linha in linhas_brutas[1:]:
                valores = [x.strip() for x in linha.split(',')]
                if len(valores) == len(cabecalho):
                    linhas.append(valores)
            if linhas:
                qas.extend(gerar_qas_tabela(cabecalho, linhas, fname, fname.split(".")[0]))
    return qas

# --- EXECUÇÃO PRINCIPAL com lógica incremental baseada em hash ---
txt_files = glob.glob("data/**/*.txt", recursive=True)
saida_dir = "data_qa"
os.makedirs(saida_dir, exist_ok=True)
hashes_path = os.path.join(saida_dir, "hashes.json")
hashes = carregar_hashes(hashes_path)

print(f"Encontrados {len(txt_files)} arquivos .txt em data/")
atualizou_hashes = False
for txt_path in txt_files:
    fname = os.path.basename(txt_path)
    out_path = os.path.join(saida_dir, fname.replace(".txt", "_qa.jsonl"))
    hash_atual = hash_arquivo(txt_path)
    # Só gera se não existir hash ou se hash mudou
    if hashes.get(fname) == hash_atual and os.path.exists(out_path):
        print(f"Pulando {fname}: hash idêntico, já processado.")
        continue
    qas = processar_arquivo_txt(txt_path)
    with open(out_path, "w", encoding="utf-8") as out:
        for qa in qas:
            out.write(json.dumps(qa, ensure_ascii=False) + "\n")
    hashes[fname] = hash_atual
    atualizou_hashes = True
    print(f"Arquivo {out_path} gerado com {len(qas)} QAs.")

if atualizou_hashes:
    salvar_hashes(hashes, hashes_path)

print(f"Pronto! Todos os QAs foram salvos/atualizados em '{saida_dir}/'")

