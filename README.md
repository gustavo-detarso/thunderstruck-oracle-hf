# Oráculo MPS — Fine-tuning, RAG e Busca Semântica em LLM

Este repositório contém uma solução completa para:
- **Geração automática de pares Pergunta/Resposta (QA)**
- **Fine-tuning eficiente de modelos LLM (LoRA)**
- **Busca semântica usando FAISS + Sentence Transformers**
- **RAG (Retrieval Augmented Generation) com interface de avaliação do usuário**

O fluxo cobre desde o processamento dos dados até o deploy do modelo treinado e a busca inteligente por contexto.

---

## 🚀 **Fluxo resumido do projeto**

1. **Geração dos dados QA**  
   - Conversão de arquivos .txt (editais, portarias, tabelas) em arquivos `.jsonl` de QA automáticos.
2. **Indexação semântica**  
   - Criação de um índice FAISS para busca rápida de contexto relevante.
3. **Fine-tuning LoRA em LLM**  
   - Treinamento do modelo com LoRA (efficient fine-tuning).
4. **Mesclagem e deploy**  
   - Merge dos adapters LoRA, push para o HuggingFace Hub, cálculo do custo de GPU.
5. **Busca RAG**  
   - Busca e geração de respostas usando o modelo treinado + contexto recuperado pelo índice.

---

## 🗃️ **Arquivos e diretórios principais**

- `preprocessing/gen_qa_chatgpt_from_txt.py`  
  Extrai QAs automáticos de arquivos `.txt`, detecta tabelas, tags e cria arquivos `.jsonl` para cada documento.

- `faiss_index.py`  
  Indexa todos os QAs em um vetor semântico (usando [Sentence Transformers](https://www.sbert.net/)), salva um índice FAISS (`faiss_chunks.index`) e o mapeamento original (`faiss_mapping.pkl`).

- `autogen-model-hf.py`  
  Script de treinamento LoRA.  
  - Usa [HuggingFace Transformers](https://huggingface.co/docs/transformers), [PEFT](https://github.com/huggingface/peft) e PyTorch.
  - Ao final, salva tanto o *adapter* LoRA (`lora_llama_finetuned/`) quanto o modelo mesclado (`merged_model/`).
  - Calcula custo do treinamento baseado no tempo e valor do droplet, consultando a cotação do dólar turismo via [AwesomeAPI](https://docs.awesomeapi.com.br/api-de-moedas).

- `rag_qa_finetuned.py`  
  Interface RAG via linha de comando:  
  - Usa embeddings para buscar o contexto mais relevante em FAISS,
  - Monta o prompt e gera resposta com o modelo fine-tuned,
  - Loga e classifica QAs para reforço futuro do dataset.

- `data_qa/`  
  Pasta com todos os arquivos `.jsonl` gerados (QAs, logs do usuário, dados validados e inválidos).

- `lora_llama_finetuned/`  
  Adapter LoRA salvo após o fine-tuning.

- `merged_model/`  
  Modelo LLM final com LoRA mesclado, pronto para deploy ou inferência stand-alone.

- `faiss_chunks.index`, `faiss_mapping.pkl`  
  Índice FAISS e mapeamento dos chunks para busca semântica.

---

## 📦 **Tecnologias e bibliotecas utilizadas**

- **[HuggingFace Transformers](https://huggingface.co/docs/transformers)**  
  Framework líder para modelos de linguagem (LLMs), NLP, geração e fine-tuning.

- **[PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)**  
  Permite treinar grandes modelos com poucos parâmetros extras, reduzindo custo e memória.  
  No projeto, usado para LoRA (Low-Rank Adaptation) — fine-tuning eficiente de LLMs.

- **[Sentence Transformers](https://www.sbert.net/)**  
  Modelos para embeddings de sentenças e busca semântica rápida.

- **[FAISS](https://github.com/facebookresearch/faiss)**  
  Biblioteca da Meta/Facebook para busca vetorial e indexação ultra-rápida (usada para o retrieval de contexto).

- **[PyTorch](https://pytorch.org/)**  
  Framework de deep learning para treinamento e inferência dos modelos.

- **[scikit-learn](https://scikit-learn.org/)**  
  Utilizado para split de datasets e manipulação básica.

- **[tqdm](https://tqdm.github.io/)**  
  Barra de progresso para loops de treinamento e indexação.

- **[TensorBoard](https://www.tensorflow.org/tensorboard)**  
  Para monitoramento visual do progresso e métricas durante o treinamento.

- **[requests](https://requests.readthedocs.io/)**  
  Usado para buscar a cotação do dólar turismo em tempo real (API AwesomeAPI).

- **[HuggingFace Hub](https://huggingface.co/docs/hub)**  
  Armazenamento, versionamento e deploy de modelos, inclusive privados.

---

## 📝 **Como rodar o projeto**

1. **Prepare seu ambiente**  
   - Instale as dependências (idealmente usando um ambiente virtual):
     ```bash
     pip install -r requirements.txt
     ```

2. **Gere QAs a partir de arquivos .txt**
   ```bash
   python preprocessing/gen_qa_chatgpt_from_txt.py
   ```

3. **Crie/atualize o índice semântico FAISS**
   ```bash
   python faiss_index.py
   ```

4. **Treine o modelo com LoRA**
   ```bash
   python autogen-model-hf.py
   ```

5. **Use a busca RAG para responder perguntas**
   ```bash
   python rag_qa_finetuned.py
   ```

---

## 🚦 **Dicas de uso/ajustes**

- Ajuste o caminho dos diretórios, nomes de modelos e parâmetros conforme sua infra.
- O fine-tuning pode ser feito em GPU local, cloud ou qualquer infra com CUDA e RAM suficiente.
- Todos os arquivos `.jsonl` em `data_qa/` podem ser incrementados manualmente ou pelo feedback dos usuários (human-in-the-loop).

---

## 🤖 **Sobre o fluxo LoRA**

- **LoRA adapter** (`lora_llama_finetuned/`): permite reusar e continuar o fine-tuning incrementalmente, ou aplicar sobre outros modelos base.
- **Modelo mesclado** (`merged_model/`): para deploy stand-alone (inference sem PEFT).

---

## 🛡️ **.gitignore sugerido**

O projeto inclui um `.gitignore` para evitar versionar:
- Ambientes virtuais,
- Dados sensíveis,
- Checkpoints/modelos grandes,
- Indexes temporários e logs.

---

## 🏷️ **Licença**

Este projeto segue a licença [MIT](LICENSE) (ou ajuste conforme sua necessidade).

---

# Qualquer dúvida ou contribuição, sinta-se à vontade para abrir uma issue ou PR!


