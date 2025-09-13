"""
Configurações do sistema LLM Judger
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configurações do Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MAX_CONCURRENT_REQUESTS = int(os.getenv("OLLAMA_MAX_CONCURRENT_REQUESTS", "4"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# Configurações de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Configurações de modelo padrão
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))

# Configurações de arquivo
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Cria diretórios se não existirem
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Templates disponíveis
TEMPLATE_TYPES = {
    "translation": "Avaliação de Tradução",
    "semantic": "Equivalência Semântica", 
    "quality": "Avaliação de Qualidade"
}

# Configurações de retry
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_BASE = int(os.getenv("RETRY_DELAY_BASE", "2"))
