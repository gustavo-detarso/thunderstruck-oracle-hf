import os
import re
from datetime import datetime
import dotenv

MAX_TOKENS = 2000
PROJETO_NOME = os.path.basename(os.path.abspath("."))

try:
    import openai
    from packaging import version
    openai_version = version.parse(openai.__version__)
except ImportError:
    openai = None
    openai_version = None

if os.path.exists(".env"):
    env = dotenv.dotenv_values(".env")
    openai_key = env.get("OPENAI_API_KEY")
else:
    openai_key = None

def ler_readme(raiz="."):
    readme_path = os.path.join(raiz, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        return ""

def coletar_lista_arquivos(raiz="."):
    arquivos = []
    for root, dirs, files in os.walk(raiz):
        # Ignora pastas comuns de build, ambiente ou repositório
        if any(x in root for x in ["logs", "debug_logs", "__pycache__", ".venv", ".git", "db", "models"]):
            continue
        for fname in files:
            if fname.lower().endswith((".py", ".r", ".toml", ".md", ".json", ".txt", ".csv", ".yml", ".yaml")):
                caminho = os.path.join(root, fname)
                arquivos.append(os.path.relpath(caminho, start=raiz))
    return arquivos

def limpar_bloco_org(texto):
    linhas = texto.strip().splitlines()
    if linhas and linhas[0].strip().startswith("```org"):
        linhas = linhas[1:]
    if linhas and linhas[-1].strip() == "```":
        linhas = linhas[:-1]
    return "\n".join(linhas).strip()

def formatar_nomes_arquivos(texto):
    # Regex para nomes de arquivos comuns
    regex = r'(?<![=\w])([\w\-/]+?\.(py|r|toml|md|json|txt|csv|yml|yaml))(?![\w=])'
    def repl(match):
        arquivo = match.group(1)
        # Não aplicar se já está entre = =
        if texto[max(0, match.start()-1)] == '=' or texto[min(len(texto)-1, match.end())] == '=':
            return arquivo
        return f'={arquivo}='
    return re.sub(regex, repl, texto)

def gerar_relatorio_ia(readme, lista_arquivos):
    if not (openai and openai_key):
        print("OpenAI/Chave não disponível. Gere um .env com sua OPENAI_API_KEY.")
        return "ERRO: IA não disponível"
    prompt = (
        f"Você é um especialista em documentação institucional de sistemas públicos de automação. "
        f"O sistema se chama {PROJETO_NOME} (nome da pasta raiz do projeto). "
        "Baseando-se no conteúdo do README.md do projeto, além da lista de arquivos presentes no diretório, gere um relatório institucional discursivo, estruturado em seções e subtítulos org-mode (use títulos como * Introdução, * Objetivos, * Dissertação, * Expectativas, * Conclusão), mas sem tópicos ou bullets, apenas texto corrido e explicativo em cada seção. "
        "Jamais utilize tópicos, bullets ou listas em nenhuma parte do texto. "
        "Jamais coloque texto em bloco de código markdown. "
        "Nunca escreva um título principal; use apenas subtítulos/seções org-mode. "
        "O relatório deve descrever a motivação do projeto, suas funções, os principais desafios enfrentados, as soluções adotadas, os objetivos institucionais, as expectativas de uso e o impacto para a gestão e para o serviço público. "
        "O relatório deve ser institucional, claro, profissional e em tom discursivo. "
        "Baseie-se também na seguinte lista de arquivos presentes no projeto para embasar a dissertação e explicar o escopo das funcionalidades implementadas:\n"
        + "\n".join(f"- {arq}" for arq in lista_arquivos)
        + "\n\nSegue o conteúdo do README.md do projeto:\n"
        + readme
        + (
            "\n\nSempre que mencionar nomes de arquivos (com extensão .py, .csv, .md, .json etc.), escreva-os entre sinais de igual, como =gen_qa_chatgptfromtxt.py=, para garantir a formatação correta na exportação LaTeX/Org-mode."
            "\n\n---\nRelatório:"
        )
    )
    try:
        if openai_version and openai_version >= version.parse("1.0.0"):
            client = openai.OpenAI(api_key=openai_key)
            resposta = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.13,
                max_tokens=MAX_TOKENS,
            )
            texto = resposta.choices[0].message.content.strip()
            return limpar_bloco_org(texto)
        else:
            openai.api_key = openai_key
            resposta = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.13,
                max_tokens=MAX_TOKENS,
            )
            texto = resposta.choices[0].message.content.strip()
            return limpar_bloco_org(texto)
    except Exception as e:
        print(f"Erro IA: {e}")
        return f"Falha IA: {e}"

def salvar_relatorio_org(texto):
    os.makedirs("docs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    caminho = f"docs/Relatorio_{PROJETO_NOME}_IA_{timestamp}.org"
    with open(caminho, "w", encoding="utf-8") as f:
        f.write('#+INCLUDE: "/home/gustavodetarso/Documentos/.share/header_mps_org/header_mps.org"\n\n')
        f.write(f'*RELATÓRIO INSTITUCIONAL – SISTEMA {PROJETO_NOME}*\n\n')
        f.write(texto)
    print(f"\nRelatório final gerado em: {caminho}")

if __name__ == "__main__":
    print(f"==> Lendo README.md e arquivos do projeto '{PROJETO_NOME}' ...")
    readme = ler_readme(".")
    lista_arquivos = coletar_lista_arquivos(".")
    print("==> Enviando para IA... (pode demorar alguns segundos/minutos)")
    relatorio_ia = gerar_relatorio_ia(readme, lista_arquivos)
    # Pós-processamento para garantir que nomes de arquivos estejam entre sinais de igual
    relatorio_ia = formatar_nomes_arquivos(relatorio_ia)
    print("==> Salvando relatório final em Org-mode...")
    salvar_relatorio_org(relatorio_ia)

