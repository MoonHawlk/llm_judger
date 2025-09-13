"""
Sistema LLM Judger - Arquivo principal modularizado
"""

import asyncio
import logging
import os
from datetime import datetime

from src.models import ModelConfig, SentencePair
from src.clients import OllamaClient
from src.judger import JudgerSystem
from src.processors import CSVProcessor
from config.settings import (
    OLLAMA_URL, OLLAMA_MAX_CONCURRENT_REQUESTS, 
    LOG_LEVEL, DEBUG_MODE, TEMPLATE_TYPES
)

# Configuração de logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


async def process_csv_dataset(judger: JudgerSystem, configs: list[ModelConfig], template_type: str):
    """Processa um dataset CSV"""
    
    print("\n=== Processamento de Dataset CSV ===\n")
    
    # Solicita arquivo CSV
    csv_path = input("Caminho para o arquivo CSV: ").strip()
    if not csv_path:
        print("❌ Caminho não fornecido!")
        return
    
    # Inicializa processador
    processor = CSVProcessor(csv_path)
    
    # Carrega CSV
    print("Carregando arquivo CSV...")
    if not processor.load_csv():
        print("❌ Erro ao carregar arquivo CSV!")
        return
    
    # Mostra colunas disponíveis
    summary = processor.get_summary()
    print(f"\nDataset carregado: {summary['total_rows']} linhas")
    print(f"Colunas disponíveis: {', '.join(summary['columns'])}")
    
    # Seleciona colunas
    print("\nSelecione as colunas:")
    for i, col in enumerate(summary['columns'], 1):
        print(f"{i}: {col}")
    
    try:
        source_choice = int(input("Coluna do texto original (número): ")) - 1
        target_choice = int(input("Coluna do texto alvo (número): ")) - 1
        
        if not (0 <= source_choice < len(summary['columns']) and 0 <= target_choice < len(summary['columns'])):
            print("❌ Números de coluna inválidos!")
            return
        
        source_col = summary['columns'][source_choice]
        target_col = summary['columns'][target_choice]
        
        if not processor.validate_columns(source_col, target_col):
            print("❌ Erro na validação das colunas!")
            return
        
        # Define idiomas
        source_lang = input("Idioma do texto original (ex: pt, en, es): ").strip() or "auto"
        target_lang = input("Idioma do texto alvo (ex: pt, en, es): ").strip() or "auto"
        processor.set_languages(source_lang, target_lang)
        
        # Converte para pares de sentenças
        print("Convertendo dados...")
        sentence_pairs = processor.get_sentence_pairs()
        
        if not sentence_pairs:
            print("❌ Nenhum par de sentenças válido encontrado!")
            return
        
        print(f"✓ {len(sentence_pairs)} pares de sentenças prontos para processamento")
        
        # Confirma processamento
        confirm = input(f"\nProcessar {len(sentence_pairs)} pares com {len(configs)} configurações? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Processamento cancelado.")
            return
        
        # Processa em lote
        print(f"\nProcessando dataset...")
        start_time = datetime.now()
        results = await judger.batch_judgment(sentence_pairs, configs, template_type)
        end_time = datetime.now()
        
        # Análise dos resultados
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n=== Resultados do Dataset ===")
        print(f"Tempo total: {(end_time - start_time).total_seconds():.2f}s")
        print(f"Sucessos: {len(successful)}/{len(results)}")
        
        if failed:
            print(f"Falhas: {len(failed)}")
        
        # Estatísticas
        if successful:
            correct_count = sum(1 for r in successful if r.is_correct)
            avg_confidence = sum(r.confidence_score for r in successful) / len(successful)
            
            print(f"\nEstatísticas:")
            print(f"- Correto: {correct_count}/{len(successful)} ({correct_count/len(successful)*100:.1f}%)")
            print(f"- Confiança média: {avg_confidence:.2f}")
        
        # Salva resultados
        output_choice = input("\nSalvar resultados? (y/n): ").strip().lower()
        if output_choice == 'y':
            output_path = input("Caminho de saída (Enter para automático): ").strip() or None
            
            if processor.save_results(successful, output_path):
                print("✓ Resultados salvos com sucesso!")
            else:
                print("❌ Erro ao salvar resultados!")
        
    except (ValueError, IndexError) as e:
        print(f"❌ Erro na seleção: {e}")
    except KeyboardInterrupt:
        print("\nProcessamento cancelado pelo usuário.")


async def main():
    """Função principal do sistema judger"""
    
    print("=== Sistema LLM Judger com Ollama ===\n")
    
    # Opção para debug
    if DEBUG_MODE:
        logging.getLogger().setLevel(logging.DEBUG)
        print("✓ Modo debug ativado")
    
    # Inicializa cliente Ollama
    client = OllamaClient(
        base_url=OLLAMA_URL, 
        max_concurrent_requests=OLLAMA_MAX_CONCURRENT_REQUESTS
    )
    judger = JudgerSystem(client)
    
    # Testa conexão com Ollama
    print("Testando conexão com Ollama...")
    if not await client.test_connection():
        print("❌ Não foi possível conectar com o Ollama.")
        print("Verifique se:")
        print("1. O Ollama está rodando (ollama serve)")
        print("2. A URL está correta (padrão: http://localhost:11434)")
        print("3. Não há firewall bloqueando a conexão")
        return
    
    # Lista modelos disponíveis
    print("Verificando modelos disponíveis...")
    available_models = await client.list_models()
    
    if not available_models:
        print("❌ Nenhum modelo encontrado.")
        print("Para instalar um modelo, use: ollama pull <nome_do_modelo>")
        print("Exemplo: ollama pull llama2")
        return
    
    print(f"✓ Modelos disponíveis: {', '.join(available_models)}")
    
    # Configuração de modelos
    configs = []
    print(f"\nModelos disponíveis:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}: {model}")
    
    while True:
        try:
            choice = input("\nEscolha o modelo (número) ou 'done': ").strip()
            
            if choice.lower() == 'done':
                break
                
            try:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(available_models):
                    model_name = available_models[model_idx]
                    
                    # Testa o modelo antes de adicionar
                    print(f"Testando modelo {model_name}...")
                    if await client.test_model(model_name):
                        instances = int(input("Número de instâncias: "))
                        configs.append(ModelConfig(name=model_name, instances=instances))
                        print(f"✓ Adicionado: {instances}x {model_name}")
                    else:
                        print(f"❌ Modelo {model_name} não está funcionando corretamente")
                else:
                    print("Número inválido!")
            except ValueError:
                print("Entrada inválida!")
                
        except KeyboardInterrupt:
            break
    
    if not configs:
        print("Nenhuma configuração definida!")
        return
    
    # Tipos de template disponíveis
    print(f"\nTipos de avaliação:")
    for key, (template_key, desc) in enumerate(TEMPLATE_TYPES.items(), 1):
        print(f"{key}: {desc}")
    
    template_choice = input("Escolha o tipo de avaliação (1-3): ").strip()
    template_keys = list(TEMPLATE_TYPES.keys())
    template_type = template_keys[int(template_choice) - 1] if template_choice.isdigit() and 1 <= int(template_choice) <= 3 else "translation"
    template_desc = TEMPLATE_TYPES[template_type]
    
    print(f"\nUsando: {template_desc}")
    print("\nConfiguração ativa:")
    for config in configs:
        print(f"- {config.instances}x {config.name}")
    
    # Loop principal de julgamento
    sentence_pairs = []
    
    while True:
        print(f"\n=== Entrada de Dados ===")
        print("Opções disponíveis:")
        print("1. Adicionar pares de sentenças manualmente")
        print("2. Processar dataset CSV")
        print("3. Processar pares adicionados")
        print("4. Sair")
        
        command = input("\nEscolha uma opção (1-4): ").strip()
        
        if command == '4' or command.lower() == 'exit':
            break
        elif command == '2':
            # Processa dataset CSV
            await process_csv_dataset(judger, configs, template_type)
        elif command == '3':
            # Processa pares adicionados manualmente
            if not sentence_pairs:
                print("Nenhum par de sentenças para processar!")
                continue
                
            print(f"\nProcessando {len(sentence_pairs)} pares com {len(configs)} configurações...")
            
            start_time = datetime.now()
            results = await judger.batch_judgment(sentence_pairs, configs, template_type)
            end_time = datetime.now()
            
            # Análise dos resultados
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            print(f"\n=== Resultados do Julgamento ===")
            print(f"Tempo total: {(end_time - start_time).total_seconds():.2f}s")
            print(f"Sucessos: {len(successful)}/{len(results)}")
            
            if failed:
                print(f"Falhas: {len(failed)}")
            
            # Agrupa resultados por par de sentenças
            pair_results = {}
            for result in successful:
                pair_id = f"{result.sentence_pair.source_text[:50]}..."
                if pair_id not in pair_results:
                    pair_results[pair_id] = []
                pair_results[pair_id].append(result)
            
            # Mostra resultados consolidados
            for pair_id, judgments in pair_results.items():
                correct_count = sum(1 for j in judgments if j.is_correct)
                avg_confidence = sum(j.confidence_score for j in judgments) / len(judgments)
                
                print(f"\n--- {pair_id} ---")
                print(f"Correto: {correct_count}/{len(judgments)} julgamentos")
                print(f"Confiança média: {avg_confidence:.2f}")
                
                # Mostra reasoning de alta confiança
                high_confidence = [j for j in judgments if j.confidence_score > 0.8]
                if high_confidence:
                    print(f"Reasoning (alta confiança): {high_confidence[0].reasoning[:200]}...")
            
            sentence_pairs = []  # Limpa para próxima rodada
        elif command == '1':
            # Adiciona pares manualmente
            try:
                source_text = input("Texto original: ").strip()
                if not source_text:
                    continue
                    
                target_text = input("Texto alvo: ").strip()
                if not target_text:
                    continue
                    
                source_lang = input("Idioma original (ex: pt, en, es): ").strip()
                target_lang = input("Idioma alvo (ex: pt, en, es): ").strip()
                context = input("Contexto (opcional): ").strip() or None
                
                pair = SentencePair(
                    source_text=source_text,
                    target_text=target_text,
                    source_language=source_lang,
                    target_language=target_lang,
                    context=context,
                    reference_id=f"pair_{len(sentence_pairs)+1}"
                )
                
                sentence_pairs.append(pair)
                print(f"✓ Adicionado par {len(sentence_pairs)}")
                
            except KeyboardInterrupt:
                print("\nOperação cancelada.")
                continue
        else:
            print("Opção inválida! Escolha 1, 2, 3 ou 4.")


if __name__ == "__main__":
    asyncio.run(main())
