#!/usr/bin/env python3
"""
Exemplo de uso do LLM Judger com datasets CSV
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

async def exemplo_csv():
    """Exemplo de processamento de dataset CSV"""
    
    print("=== Exemplo de Processamento CSV ===\n")
    
    # Inicializa cliente
    client = OllamaClient(base_url="http://localhost:11434")
    
    # Testa conexão
    if not await client.test_connection():
        print("❌ Ollama não está rodando!")
        return False
    
    # Lista modelos
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível!")
        return False
    
    print(f"✓ Modelos disponíveis: {models}")
    
    # Usa o primeiro modelo
    model_name = models[0]
    print(f"Usando modelo: {model_name}")
    
    # Inicializa processador CSV
    csv_path = "exemplo_dataset.csv"
    processor = CSVProcessor(csv_path)
    
    # Carrega CSV
    print(f"\nCarregando {csv_path}...")
    if not processor.load_csv():
        print("❌ Erro ao carregar CSV!")
        return False
    
    # Configura colunas (assumindo que o CSV tem as colunas padrão)
    summary = processor.get_summary()
    print(f"Colunas encontradas: {summary['columns']}")
    
    # Configura automaticamente se possível
    if 'texto_original' in summary['columns'] and 'texto_traduzido' in summary['columns']:
        processor.validate_columns('texto_original', 'texto_traduzido')
        processor.set_languages('en', 'pt')
    else:
        print("❌ Colunas esperadas não encontradas!")
        return False
    
    # Converte para pares de sentenças
    sentence_pairs = processor.get_sentence_pairs()
    print(f"✓ {len(sentence_pairs)} pares de sentenças carregados")
    
    # Configura modelo
    configs = [ModelConfig(name=model_name, instances=1)]
    
    # Processa
    print("\nProcessando dataset...")
    judger = JudgerSystem(client)
    results = await judger.batch_judgment(
        sentence_pairs=sentence_pairs,
        configs=configs,
        template_type="translation"
    )
    
    # Análise dos resultados
    successful = [r for r in results if r.success]
    print(f"\nResultados: {len(successful)}/{len(results)} sucessos")
    
    if successful:
        correct_count = sum(1 for r in successful if r.is_correct)
        avg_confidence = sum(r.confidence_score for r in successful) / len(successful)
        
        print(f"Estatísticas:")
        print(f"- Correto: {correct_count}/{len(successful)} ({correct_count/len(successful)*100:.1f}%)")
        print(f"- Confiança média: {avg_confidence:.2f}")
        
        # Salva resultados
        output_path = "exemplo_dataset_resultados.csv"
        if processor.save_results(successful, output_path):
            print(f"✓ Resultados salvos em: {output_path}")
        else:
            print("❌ Erro ao salvar resultados!")
    
    return len(successful) > 0

if __name__ == "__main__":
    success = asyncio.run(exemplo_csv())
    if success:
        print("\n✅ Exemplo CSV executado com sucesso!")
    else:
        print("\n❌ Exemplo CSV falhou!")
