#!/usr/bin/env python3
"""
Teste de imports para verificar se a estrutura modular está funcionando
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório atual ao path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Testar imports
    from src.judger import JudgerSystem
    from src.models import ModelConfig, SentencePair
    from src.clients import OllamaClient
    from src.processors import CSVProcessor
    from src.templates import PromptTemplate
    from src.utils import setup_logging
    from config.settings import OLLAMA_URL
    
    print("✅ Todos os imports funcionaram corretamente!")
    print(f"✅ Ollama URL configurada: {OLLAMA_URL}")
    
    # Testar criação de objetos básicos
    config = ModelConfig(name="test", instances=1)
    pair = SentencePair("Hello", "Olá", "en", "pt")
    
    print("✅ Objetos criados com sucesso!")
    print(f"✅ ModelConfig: {config.name} x{config.instances}")
    print(f"✅ SentencePair: {pair.source_text} -> {pair.target_text}")
    
except ImportError as e:
    print(f"❌ Erro de import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erro geral: {e}")
    sys.exit(1)

print("\n🎉 Estrutura modular está funcionando perfeitamente!")
