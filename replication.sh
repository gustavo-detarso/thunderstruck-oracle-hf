#!/usr/bin/env bash
set -e

# Esse script:
# 1) Desabilita o módulo nouveau
# 2) Habilita contrib, non-free, non-free-firmware no Debian Bookworm
# 3) Atualiza APT e faz autoremove
# 4) Instala CUDA (driver + toolkit) via metapacote oficial
# 5) Instala dependências do sistema para pyenv
# 6) Clona e configura pyenv
# 7) Instala Python 3.11 via pyenv, cria venv e instala requirements
# 8) Reinicia a máquina ao final

# 0) Configurações
DEBIAN_CODENAME="bookworm"
PYTHON_VERSION="3.11.13"
PROJECT_DIR="$(pwd)"
REQUIREMENTS_FILE="requirements.txt"

echo "=== 1) Desabilitando módulo nouveau ==="
sudo tee /etc/modprobe.d/disable-nouveau.conf > /dev/null <<EOF
blacklist nouveau
options nouveau modeset=0
EOF
sudo update-initramfs -u

echo "=== 2) Ajustando repositórios non-free/contrib/non-free-firmware ==="
sudo sed -i.bak -E \
  "/^deb .* ${DEBIAN_CODENAME}(-security|-updates|-backports)? .* main/ \
   s/main/main contrib non-free non-free-firmware/" \
  /etc/apt/sources.list

grep -q "${DEBIAN_CODENAME}-security" /etc/apt/sources.list || \
  echo "deb http://deb.debian.org/debian-security ${DEBIAN_CODENAME}-security main contrib non-free non-free-firmware" \
    | sudo tee -a /etc/apt/sources.list
grep -q "${DEBIAN_CODENAME}-backports" /etc/apt/sources.list || \
  echo "deb http://deb.debian.org/debian ${DEBIAN_CODENAME}-backports main contrib non-free non-free-firmware" \
    | sudo tee -a /etc/apt/sources.list

echo "=== 3) Atualizando APT ==="
sudo apt update

echo "=== 3.1) Removendo pacotes obsoletos ==="
sudo apt autoremove -y

echo "=== 4) Instalando CUDA (driver + toolkit) ==="
wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda

echo "=== 5) Instalando dependências do pyenv e build de Python ==="
sudo apt install -y \
  build-essential curl git libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget llvm libncurses5-dev \
  libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
  libxml2-dev libxmlsec1-dev

echo "=== 6) Instalando pyenv ==="
if [ ! -d "$HOME/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git ~/.pyenv
fi

echo "=== 7) Configurando pyenv no ~/.bashrc ==="
if ! grep -q 'PYENV_ROOT' ~/.bashrc; then
  cat << 'EOF' >> ~/.bashrc

# >>> pyenv setup >>>
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"
fi
# <<< end pyenv setup <<<
EOF
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

echo "=== 8) Instalando Python $PYTHON_VERSION via pyenv ==="
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
  pyenv install "$PYTHON_VERSION"
fi
pyenv local "$PYTHON_VERSION"

echo "=== 9) Criando e ativando virtualenv (.venv) ==="
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== 10) Instalando dependências Python ==="
pip install --upgrade pip setuptools wheel
if [ -f "$REQUIREMENTS_FILE" ]; then
  pip install -r "$REQUIREMENTS_FILE"
else
  echo "⚠️  $REQUIREMENTS_FILE não encontrado; pulando."
fi

echo "=== 10.1) Autenticando no HuggingFace CLI (via .env) ==="
if grep -q HUGGINGFACE_TOKEN "$PROJECT_DIR/.env"; then
  export $(grep HUGGINGFACE_TOKEN "$PROJECT_DIR/.env")
  if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "⚠️  HUGGINGFACE_TOKEN não encontrado no .env ou está vazio!"
  else
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
    echo "Login HuggingFace realizado!"
  fi
else
  echo "⚠️  HUGGINGFACE_TOKEN não encontrado no .env. Skipping HF login."
fi

echo
echo "✅ Ambiente configurado com sucesso!"
echo "⚠️  Um reboot é necessário para desativar o nouveau e ativar o driver NVIDIA."
sudo reboot

