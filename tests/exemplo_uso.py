#!/usr/bin/env python3
"""
Exemplo de uso do LLM Judger com Ollama
Este script demonstra como usar o sistema programaticamente
"""
import sys
import os
from pathlib import Path

# Adicionar o diretório pai ao path para importar src
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import asyncio
from src.judger import JudgerSystem
from src.models import ModelConfig, SentencePair
from src.clients import OllamaClient
from src.processors import CSVProcessor

async def exemplo_basico():
    """Exemplo básico de uso do sistema"""
    
    print("=== Exemplo de Uso do LLM Judger ===\n")
    
    # Inicializa cliente
    client = OllamaClient(base_url="http://localhost:11434")
    judger = JudgerSystem(client)
    
    # Testa conexão
    if not await client.test_connection():
        print("❌ Ollama não está rodando!")
        return
    
    # Lista modelos
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível!")
        return
    
    print(f"Modelos disponíveis: {models}")
    
    # Usa o primeiro modelo disponível
    model_name = models[0]
    print(f"Usando modelo: {model_name}")
    
    # Testa o modelo
    if not await client.test_model(model_name):
        print(f"❌ Modelo {model_name} não está funcionando!")
        return
    
    # Cria um par de sentenças para testar
    sentence_pair = SentencePair(
        source_text="Hello, how are you?",
        target_text="Olá, como você está?",
        source_language="en",
        target_language="pt",
        context="Saudação informal"
    )
    
    # Faz o julgamento
    print("\nFazendo julgamento...")
    result = await judger.judge_sentence_pair(
        sentence_pair=sentence_pair,
        model=model_name,
        template_type="translation"
    )
    
    # Mostra resultado
    print(f"\n=== Resultado ===")
    print(f"Modelo: {result.model}")
    print(f"Sucesso: {result.success}")
    print(f"Correto: {result.is_correct}")
    print(f"Confiança: {result.confidence_score:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    if result.error:
        print(f"Erro: {result.error}")

async def exemplo_batch():
    """Exemplo de processamento em lote"""
    
    print("\n=== Exemplo de Processamento em Lote ===\n")
    
    client = OllamaClient(base_url="http://localhost:11434")
    judger = JudgerSystem(client)
    
    if not await client.test_connection():
        print("❌ Ollama não está rodando!")
        return
    
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível!")
        return
    
    # Cria múltiplos pares de sentenças
    sentence_pairs = [
        SentencePair(
            source_text="Good morning!",
            target_text="Bom dia!",
            source_language="en",
            target_language="pt"
        ),
        SentencePair(
            source_text="Thank you very much.",
            target_text="Muito obrigado.",
            source_language="en",
            target_language="pt"
        ),
        SentencePair(
            source_text="I love you.",
            target_text="Eu te amo.",
            source_language="en",
            target_language="pt"
        )
    ]
    
    # Configura modelo
    configs = [ModelConfig(name=models[0], instances=2)]
    
    # Processa em lote
    print(f"Processando {len(sentence_pairs)} pares com {configs[0].instances} instâncias...")
    results = await judger.batch_judgment(
        sentence_pairs=sentence_pairs,
        configs=configs,
        template_type="translation"
    )
    
    # Analisa resultados
    successful = [r for r in results if r.success]
    print(f"\nResultados: {len(successful)}/{len(results)} sucessos")
    
    for i, result in enumerate(successful):
        print(f"\n--- Par {i+1} ---")
        print(f"Correto: {result.is_correct}")
        print(f"Confiança: {result.confidence_score:.2f}")
        print(f"Reasoning: {result.reasoning[:100]}...")

if __name__ == "__main__":
    print("Escolha o exemplo:")
    print("1: Exemplo básico")
    print("2: Exemplo em lote")
    
    choice = input("Digite 1 ou 2: ").strip()
    
    if choice == "1":
        asyncio.run(exemplo_basico())
    elif choice == "2":
        asyncio.run(exemplo_batch())
    else:
        print("Opção inválida!")
