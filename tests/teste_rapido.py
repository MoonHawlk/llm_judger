#!/usr/bin/env python3
"""
Teste rápido do sistema LLM Judger
"""
import sys
import os
from pathlib import Path

# Adicionar o diretório pai ao path para importar src
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import asyncio
import logging
from src.judger import JudgerSystem
from src.models import ModelConfig, SentencePair
from src.clients import OllamaClient
from src.processors import CSVProcessor

# Configurar logging para debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def teste_rapido():
    """Teste rápido do sistema"""
    
    print("=== Teste Rápido do LLM Judger ===\n")
    
    # Inicializa cliente
    client = OllamaClient(base_url="http://localhost:11434")
    
    # Testa conexão
    print("1. Testando conexão com Ollama...")
    if not await client.test_connection():
        print("❌ Ollama não está rodando!")
        return False
    
    # Lista modelos
    print("2. Listando modelos...")
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível!")
        return False
    
    print(f"✓ Modelos encontrados: {models}")
    
    # Testa primeiro modelo
    model_name = models[0]
    print(f"3. Testando modelo {model_name}...")
    if not await client.test_model(model_name):
        print(f"❌ Modelo {model_name} não está funcionando!")
        return False
    
    # Testa julgamento
    print("4. Testando julgamento...")
    judger = JudgerSystem(client)
    
    sentence_pair = SentencePair(
        source_text="Hello, how are you?",
        target_text="Olá, como você está?",
        source_language="en",
        target_language="pt"
    )
    
    result = await judger.judge_sentence_pair(
        sentence_pair=sentence_pair,
        model=model_name,
        template_type="translation"
    )
    
    print(f"✓ Julgamento concluído!")
    print(f"  - Sucesso: {result.success}")
    print(f"  - Correto: {result.is_correct}")
    print(f"  - Confiança: {result.confidence_score:.2f}")
    print(f"  - Reasoning: {result.reasoning[:100]}...")
    
    if result.error:
        print(f"  - Erro: {result.error}")
    
    return result.success

if __name__ == "__main__":
    success = asyncio.run(teste_rapido())
    if success:
        print("\n✅ Teste passou! O sistema está funcionando corretamente.")
    else:
        print("\n❌ Teste falhou! Verifique os logs acima.")
