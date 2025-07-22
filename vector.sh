#!/usr/bin/env bash

# Script para copiar todo o diretório atual para um servidor remoto via SSH,
# criando antes a pasta remota (por padrão ~/thunderstruck-oracle-hf).
# Se KEY_FILE ficar em branco, usa o SSH Agent / chaves padrão (~/.ssh/id_*).

# Defaults
DEFAULT_IP="165.227.34.149"
DEFAULT_USER="root"
DEFAULT_REMOTE_DIR="~/thunderstruck-oracle-hf"

# 1) Lê parâmetros
read -p "Endereço IP do droplet [${DEFAULT_IP}]: " REMOTE_IP
REMOTE_IP="${REMOTE_IP:-$DEFAULT_IP}"

read -p "Usuário SSH [${DEFAULT_USER}]: " SSH_USER
SSH_USER="${SSH_USER:-$DEFAULT_USER}"

read -p "Diretório remoto onde criar thunderstruck-oracle-hf [${DEFAULT_REMOTE_DIR}]: " REMOTE_DIR
REMOTE_DIR="${REMOTE_DIR:-$DEFAULT_REMOTE_DIR}"

read -p "Caminho para chave privada SSH (ENTER para usar agent/default): " KEY_FILE

# Prepara o parâmetro de chave, se fornecido
SSH_KEY_OPTION=""
if [[ -n "$KEY_FILE" ]]; then
  if [[ ! -f "$KEY_FILE" ]]; then
    echo "❌ Arquivo de chave não encontrado: $KEY_FILE"
    exit 1
  fi
  if [[ "$KEY_FILE" == *.pub ]]; then
    echo "❌ Você forneceu um .pub — use a chave privada (sem .pub)"
    exit 1
  fi
  SSH_KEY_OPTION="-i $KEY_FILE"
fi

# 2) Testa conexão SSH
echo -e "\n🔍 Testando conexão SSH..."
ssh $SSH_KEY_OPTION -o BatchMode=yes -o StrictHostKeyChecking=no \
    "${SSH_USER}@${REMOTE_IP}" "echo 'Autenticado!'" >/dev/null 2>&1

if [ $? -ne 0 ]; then
  echo "❌ Falha na autenticação SSH."
  echo "   - Se estiver usando agent, certifique-se de ter rodado 'ssh-add'."
  echo "   - Se não, forneça o caminho da chave privada."
  exit 1
fi
echo "✅ SSH OK"

# 3) Cria diretório remoto
echo -e "\n→ Criando diretório remoto ${REMOTE_DIR}"
ssh $SSH_KEY_OPTION -o StrictHostKeyChecking=no \
    "${SSH_USER}@${REMOTE_IP}" "mkdir -p ${REMOTE_DIR}" || {
  echo "❌ Não foi possível criar ${REMOTE_DIR} no host remoto."
  exit 1
}

# 4) Sincroniza
echo -e "\n→ Sincronizando $(pwd)/ → ${SSH_USER}@${REMOTE_IP}:${REMOTE_DIR}/"
if command -v rsync &> /dev/null; then
  rsync -avz -e "ssh $SSH_KEY_OPTION -o StrictHostKeyChecking=no" \
    ./ "${SSH_USER}@${REMOTE_IP}:${REMOTE_DIR}/" || {
      echo "❌ Erro no rsync"; exit 1
    }
else
  echo "rsync não encontrado, usando scp recursivo"
  scp $SSH_KEY_OPTION -r "$(pwd)/." "${SSH_USER}@${REMOTE_IP}:${REMOTE_DIR}/" || {
    echo "❌ Erro no scp"; exit 1
  }
fi

echo -e "\n✅ Cópia concluída com sucesso!"

