#!/usr/bin/env python3
"""
Teste simples e direto para funcionalidades CSV
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def teste_simples_csv():
    """Teste simples do processamento CSV"""
    
    print("=== Teste Simples CSV ===\n")
    
    # 1. Setup básico
    print("1. Configurando cliente...")
    client = OllamaClient(base_url="http://localhost:11434")
    
    if not await client.test_connection():
        print("❌ Ollama não está rodando!")
        return False
    
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível!")
        return False
    
    model_name = models[0]
    print(f"✓ Usando modelo: {model_name}")
    
    # 2. Testa processamento do arquivo exemplo
    print("\n2. Processando arquivo exemplo...")
    processor = CSVProcessor("exemplo_dataset.csv")
    
    if not processor.load_csv():
        print("❌ Erro ao carregar exemplo_dataset.csv!")
        return False
    
    # Configura colunas
    processor.validate_columns('texto_original', 'texto_traduzido')
    processor.set_languages('en', 'pt')
    
    # Converte para pares
    sentence_pairs = processor.get_sentence_pairs()
    print(f"✓ {len(sentence_pairs)} pares carregados")
    
    # 3. Testa julgamento
    print("\n3. Testando julgamento...")
    judger = JudgerSystem(client)
    configs = [ModelConfig(name=model_name, instances=1)]
    
    # Processa apenas os primeiros 3 pares para teste rápido
    test_pairs = sentence_pairs[:3]
    print(f"Processando {len(test_pairs)} pares de teste...")
    
    results = await judger.batch_judgment(
        sentence_pairs=test_pairs,
        configs=configs,
        template_type="translation"
    )
    
    successful = [r for r in results if r.success]
    print(f"✓ {len(successful)}/{len(results)} julgamentos bem-sucedidos")
    
    # 4. Testa salvamento
    print("\n4. Testando salvamento...")
    if successful:
        output_file = "teste_resultados.csv"
        if processor.save_results(successful, output_file):
            print(f"✓ Resultados salvos em: {output_file}")
            
            # Mostra resumo
            correct_count = sum(1 for r in successful if r.is_correct)
            avg_confidence = sum(r.confidence_score for r in successful) / len(successful)
            
            print(f"\nResumo dos resultados:")
            print(f"  - Correto: {correct_count}/{len(successful)}")
            print(f"  - Confiança média: {avg_confidence:.2f}")
            
            return True
        else:
            print("❌ Erro ao salvar resultados!")
            return False
    else:
        print("❌ Nenhum resultado para salvar!")
        return False

async def main():
    """Função principal"""
    sucesso = await teste_simples_csv()
    
    if sucesso:
        print("\n✅ TESTE CSV SIMPLES PASSOU!")
        print("O sistema está funcionando corretamente.")
    else:
        print("\n❌ TESTE CSV SIMPLES FALHOU!")
        print("Verifique os logs acima.")

if __name__ == "__main__":
    asyncio.run(main())
