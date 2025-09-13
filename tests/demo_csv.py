#!/usr/bin/env python3
"""
Demonstração completa do sistema CSV
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_completa():
    """Demonstração completa do sistema"""
    
    print("🎬 DEMONSTRAÇÃO COMPLETA DO SISTEMA CSV\n")
    
    # 1. Setup
    print("1️⃣ Configurando sistema...")
    client = OllamaClient(base_url="http://localhost:11434")
    
    if not await client.test_connection():
        print("❌ Ollama não está rodando! Execute: ollama serve")
        return
    
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível! Execute: ollama pull llama2")
        return
    
    model_name = models[0]
    print(f"✓ Usando modelo: {model_name}")
    
    # 2. Carrega dataset
    print("\n2️⃣ Carregando dataset...")
    # Caminho para o arquivo CSV
    csv_path = parent_dir / "data" / "exemplo_dataset.csv"
    processor = CSVProcessor(str(csv_path))
    
    if not processor.load_csv():
        print("❌ Arquivo exemplo_dataset.csv não encontrado!")
        return
    
    # Configura
    processor.validate_columns('texto_original', 'texto_traduzido')
    processor.set_languages('en', 'pt')
    
    sentence_pairs = processor.get_sentence_pairs()
    print(f"✓ {len(sentence_pairs)} pares carregados")
    
    # Mostra alguns exemplos
    print("\n📋 Exemplos do dataset:")
    for i, pair in enumerate(sentence_pairs[:3]):
        print(f"  {i+1}. '{pair.source_text}' → '{pair.target_text}'")
    
    # 3. Processa
    print(f"\n3️⃣ Processando com {model_name}...")
    judger = JudgerSystem(client)
    configs = [ModelConfig(name=model_name, instances=1)]
    
    results = await judger.batch_judgment(
        sentence_pairs=sentence_pairs,
        configs=configs,
        template_type="translation"
    )
    
    successful = [r for r in results if r.success]
    print(f"✓ {len(successful)}/{len(results)} julgamentos concluídos")
    
    # 4. Análise
    print("\n4️⃣ Analisando resultados...")
    if successful:
        correct_count = sum(1 for r in successful if r.is_correct)
        avg_confidence = sum(r.confidence_score for r in successful) / len(successful)
        
        print(f"📊 Estatísticas:")
        print(f"  • Correto: {correct_count}/{len(successful)} ({correct_count/len(successful)*100:.1f}%)")
        print(f"  • Confiança média: {avg_confidence:.2f}")
        
        # Mostra alguns resultados
        print(f"\n🔍 Exemplos de julgamentos:")
        for i, result in enumerate(successful[:3]):
            status = "✅" if result.is_correct else "❌"
            print(f"  {i+1}. {status} {result.confidence_score:.2f}")
            print(f"     '{result.sentence_pair.source_text}' → '{result.sentence_pair.target_text}'")
            print(f"     💭 {result.reasoning[:80]}...")
            print()
    
    # 5. Salva resultados
    print("5️⃣ Salvando resultados...")
    output_file = "demo_resultados.csv"
    
    if processor.save_results(successful, output_file):
        print(f"✓ Resultados salvos em: {output_file}")
        
        # Mostra preview do arquivo
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"\n📄 Preview do arquivo de resultados:")
        print(df[['texto_original', 'texto_traduzido', 'resultado', 'confianca']].head())
        
        print(f"\n🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"📁 Arquivo de resultados: {output_file}")
        print(f"📈 {len(successful)} julgamentos processados")
        
    else:
        print("❌ Erro ao salvar resultados!")

async def main():
    """Função principal"""
    try:
        await demo_completa()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstração interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro durante demonstração: {e}")
        logger.exception("Erro detalhado:")

if __name__ == "__main__":
    asyncio.run(main())
