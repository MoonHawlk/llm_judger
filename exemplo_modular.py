"""
Exemplo de uso do sistema LLM Judger modularizado
"""

import asyncio
import logging
from src.models import ModelConfig, SentencePair
from src.clients import OllamaClient
from src.judger import JudgerSystem
from src.processors import CSVProcessor
from src.utils import print_results_summary, print_detailed_results
from config.settings import OLLAMA_URL, OLLAMA_MAX_CONCURRENT_REQUESTS

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def exemplo_julgamento_manual():
    """Exemplo de julgamento manual de pares de sentenças"""
    
    print("=== Exemplo: Julgamento Manual ===\n")
    
    # Configurar cliente e sistema
    client = OllamaClient(
        base_url=OLLAMA_URL,
        max_concurrent_requests=OLLAMA_MAX_CONCURRENT_REQUESTS
    )
    judger = JudgerSystem(client)
    
    # Testar conexão
    if not await client.test_connection():
        print("❌ Não foi possível conectar com Ollama")
        return
    
    # Listar modelos disponíveis
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível")
        return
    
    print(f"Modelos disponíveis: {', '.join(models)}")
    
    # Usar o primeiro modelo disponível
    model_name = models[0]
    print(f"Usando modelo: {model_name}")
    
    # Criar pares de sentenças para teste
    sentence_pairs = [
        SentencePair(
            source_text="Hello, how are you?",
            target_text="Olá, como você está?",
            source_language="en",
            target_language="pt",
            reference_id="test_1"
        ),
        SentencePair(
            source_text="The weather is nice today.",
            target_text="O tempo está bom hoje.",
            source_language="en",
            target_language="pt",
            reference_id="test_2"
        ),
        SentencePair(
            source_text="I love programming.",
            target_text="Eu amo programar.",
            source_language="en",
            target_language="pt",
            reference_id="test_3"
        )
    ]
    
    # Configurar modelo
    config = ModelConfig(name=model_name, instances=1)
    
    # Fazer julgamentos
    print(f"\nProcessando {len(sentence_pairs)} pares de sentenças...")
    results = await judger.batch_judgment(sentence_pairs, [config], "translation")
    
    # Mostrar resultados
    print_results_summary(results)
    print_detailed_results(results)


async def exemplo_processamento_csv():
    """Exemplo de processamento de arquivo CSV"""
    
    print("\n=== Exemplo: Processamento CSV ===\n")
    
    # Caminho para o arquivo CSV de exemplo
    csv_path = "data/exemplo_dataset.csv"
    
    # Configurar processador
    processor = CSVProcessor(csv_path)
    
    # Carregar CSV
    if not processor.load_csv():
        print("❌ Erro ao carregar CSV")
        return
    
    # Mostrar informações do dataset
    summary = processor.get_summary()
    print(f"Dataset carregado: {summary['total_rows']} linhas")
    print(f"Colunas: {', '.join(summary['columns'])}")
    
    # Configurar colunas (assumindo que existem colunas 'source' e 'target')
    if 'source' in summary['columns'] and 'target' in summary['columns']:
        processor.validate_columns('source', 'target')
        processor.set_languages('en', 'pt')
        
        # Extrair pares de sentenças
        sentence_pairs = processor.get_sentence_pairs()
        print(f"Extraídos {len(sentence_pairs)} pares válidos")
        
        if sentence_pairs:
            # Configurar cliente e sistema
            client = OllamaClient(base_url=OLLAMA_URL)
            judger = JudgerSystem(client)
            
            # Testar conexão
            if await client.test_connection():
                models = await client.list_models()
                if models:
                    model_name = models[0]
                    config = ModelConfig(name=model_name, instances=1)
                    
                    # Processar apenas os primeiros 2 pares para exemplo
                    limited_pairs = sentence_pairs[:2]
                    print(f"Processando {len(limited_pairs)} pares...")
                    
                    results = await judger.batch_judgment(limited_pairs, [config], "translation")
                    print_results_summary(results)
                    
                    # Salvar resultados
                    output_path = "data/exemplo_resultados.csv"
                    if processor.save_results(results, output_path):
                        print(f"✓ Resultados salvos em: {output_path}")
    else:
        print("❌ Colunas 'source' e 'target' não encontradas no CSV")


async def exemplo_diferentes_templates():
    """Exemplo usando diferentes tipos de templates"""
    
    print("\n=== Exemplo: Diferentes Templates ===\n")
    
    # Configurar cliente e sistema
    client = OllamaClient(base_url=OLLAMA_URL)
    judger = JudgerSystem(client)
    
    if not await client.test_connection():
        print("❌ Não foi possível conectar com Ollama")
        return
    
    models = await client.list_models()
    if not models:
        print("❌ Nenhum modelo disponível")
        return
    
    model_name = models[0]
    config = ModelConfig(name=model_name, instances=1)
    
    # Par de sentenças para teste
    sentence_pair = SentencePair(
        source_text="The cat is sleeping on the sofa.",
        target_text="O gato está dormindo no sofá.",
        source_language="en",
        target_language="pt",
        reference_id="template_test"
    )
    
    # Testar diferentes tipos de templates
    template_types = ["translation", "semantic", "quality"]
    
    for template_type in template_types:
        print(f"\n--- Testando template: {template_type} ---")
        
        result = await judger.judge_sentence_pair(sentence_pair, model_name, template_type)
        
        if result.success:
            print(f"Correto: {result.is_correct}")
            print(f"Confiança: {result.confidence_score:.2f}")
            print(f"Reasoning: {result.reasoning[:100]}...")
        else:
            print(f"❌ Erro: {result.error}")


async def main():
    """Função principal do exemplo"""
    
    print("=== Exemplos do Sistema LLM Judger Modularizado ===\n")
    
    try:
        # Exemplo 1: Julgamento manual
        await exemplo_julgamento_manual()
        
        # Exemplo 2: Processamento CSV
        await exemplo_processamento_csv()
        
        # Exemplo 3: Diferentes templates
        await exemplo_diferentes_templates()
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        print(f"❌ Erro: {e}")
    
    print("\n=== Exemplos concluídos ===")


if __name__ == "__main__":
    asyncio.run(main())
