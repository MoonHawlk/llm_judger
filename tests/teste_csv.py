#!/usr/bin/env python3
"""
Script de teste completo para funcionalidades CSV do LLM Judger
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
import os
import pandas as pd
from pathlib import Path
from src.judger import JudgerSystem
from src.models import ModelConfig, SentencePair
from src.clients import OllamaClient
from src.processors import CSVProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TesteCSV:
    """Classe para testar funcionalidades CSV"""
    
    def __init__(self):
        self.client = None
        self.judger = None
        self.test_csv_path = "teste_dataset.csv"
        self.result_csv_path = "teste_dataset_resultados.csv"
        
    async def setup(self):
        """Configura o ambiente de teste"""
        print("=== Configurando Ambiente de Teste ===\n")
        
        # Inicializa cliente Ollama
        self.client = OllamaClient(base_url="http://localhost:11434")
        
        # Testa conexão
        if not await self.client.test_connection():
            print("❌ Ollama não está rodando!")
            return False
        
        # Lista modelos
        models = await self.client.list_models()
        if not models:
            print("❌ Nenhum modelo disponível!")
            return False
        
        print(f"✓ Modelos disponíveis: {models}")
        
        # Usa o primeiro modelo
        model_name = models[0]
        print(f"✓ Usando modelo: {model_name}")
        
        # Testa o modelo
        if not await self.client.test_model(model_name):
            print(f"❌ Modelo {model_name} não está funcionando!")
            return False
        
        # Inicializa judger
        self.judger = JudgerSystem(self.client)
        
        return True
    
    def criar_dataset_teste(self):
        """Cria um dataset de teste"""
        print("\n=== Criando Dataset de Teste ===\n")
        
        # Dados de teste com diferentes cenários
        dados_teste = [
            # Traduções corretas
            ("Hello, how are you?", "Olá, como você está?", "correto"),
            ("Good morning!", "Bom dia!", "correto"),
            ("Thank you very much.", "Muito obrigado.", "correto"),
            ("I love you.", "Eu te amo.", "correto"),
            
            # Traduções incorretas
            ("What time is it?", "Como você está?", "incorreto"),
            ("Where is the bathroom?", "Bom dia!", "incorreto"),
            ("How much does this cost?", "Eu te amo.", "incorreto"),
            
            # Traduções parciais/ambiguas
            ("See you later!", "Até mais tarde!", "parcial"),
            ("Have a nice day!", "Tenha um bom dia!", "correto"),
            
            # Casos especiais
            ("", "Texto vazio", "vazio"),
            ("Texto vazio", "", "vazio"),
        ]
        
        # Cria DataFrame
        df = pd.DataFrame(dados_teste, columns=['texto_original', 'texto_traduzido', 'esperado'])
        
        # Salva CSV
        df.to_csv(self.test_csv_path, index=False, encoding='utf-8')
        print(f"✓ Dataset de teste criado: {self.test_csv_path}")
        print(f"  - {len(df)} linhas")
        print(f"  - Colunas: {list(df.columns)}")
        
        return True
    
    def testar_carregamento_csv(self):
        """Testa carregamento do CSV"""
        print("\n=== Testando Carregamento CSV ===\n")
        
        processor = CSVProcessor(self.test_csv_path)
        
        # Testa carregamento
        if not processor.load_csv():
            print("❌ Erro ao carregar CSV!")
            return False
        
        # Testa validação de colunas
        if not processor.validate_columns('texto_original', 'texto_traduzido'):
            print("❌ Erro na validação de colunas!")
            return False
        
        # Testa configuração de idiomas
        processor.set_languages('en', 'pt')
        
        # Testa conversão para pares
        sentence_pairs = processor.get_sentence_pairs()
        print(f"✓ {len(sentence_pairs)} pares de sentenças carregados")
        
        # Mostra alguns exemplos
        for i, pair in enumerate(sentence_pairs[:3]):
            print(f"  {i+1}. '{pair.source_text}' -> '{pair.target_text}'")
        
        return processor, sentence_pairs
    
    async def testar_julgamento(self, sentence_pairs):
        """Testa o julgamento das sentenças"""
        print("\n=== Testando Julgamento ===\n")
        
        # Configura modelo
        models = await self.client.list_models()
        model_name = models[0]
        configs = [ModelConfig(name=model_name, instances=1)]
        
        print(f"Processando {len(sentence_pairs)} pares...")
        
        # Processa em lote
        results = await self.judger.batch_judgment(
            sentence_pairs=sentence_pairs,
            configs=configs,
            template_type="translation"
        )
        
        # Análise dos resultados
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✓ Resultados: {len(successful)}/{len(results)} sucessos")
        
        if failed:
            print(f"❌ Falhas: {len(failed)}")
            for fail in failed:
                print(f"  - {fail.error}")
        
        # Estatísticas
        if successful:
            correct_count = sum(1 for r in successful if r.is_correct)
            avg_confidence = sum(r.confidence_score for r in successful) / len(successful)
            
            print(f"\nEstatísticas:")
            print(f"  - Correto: {correct_count}/{len(successful)} ({correct_count/len(successful)*100:.1f}%)")
            print(f"  - Confiança média: {avg_confidence:.2f}")
            
            # Mostra alguns resultados
            print(f"\nExemplos de resultados:")
            for i, result in enumerate(successful[:3]):
                status = "✓" if result.is_correct else "✗"
                print(f"  {i+1}. {status} {result.confidence_score:.2f} - {result.reasoning[:50]}...")
        
        return successful
    
    def testar_salvamento_resultados(self, processor, results):
        """Testa salvamento dos resultados"""
        print("\n=== Testando Salvamento de Resultados ===\n")
        
        # Salva resultados
        if processor.save_results(results, self.result_csv_path):
            print(f"✓ Resultados salvos em: {self.result_csv_path}")
            
            # Verifica arquivo criado
            if Path(self.result_csv_path).exists():
                result_df = pd.read_csv(self.result_csv_path)
                print(f"✓ Arquivo verificado: {len(result_df)} linhas")
                print(f"  - Colunas: {list(result_df.columns)}")
                
                # Mostra algumas linhas
                print(f"\nPrimeiras linhas do resultado:")
                print(result_df[['texto_original', 'texto_traduzido', 'resultado', 'confianca']].head())
                
                return True
            else:
                print("❌ Arquivo de resultado não foi criado!")
                return False
        else:
            print("❌ Erro ao salvar resultados!")
            return False
    
    def testar_casos_especiais(self):
        """Testa casos especiais"""
        print("\n=== Testando Casos Especiais ===\n")
        
        # Testa CSV com diferentes encodings
        print("1. Testando diferentes encodings...")
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(self.test_csv_path, encoding=encoding)
                print(f"  ✓ {encoding}: {len(df)} linhas")
            except Exception as e:
                print(f"  ✗ {encoding}: {e}")
        
        # Testa CSV com colunas diferentes
        print("\n2. Testando validação de colunas...")
        processor = CSVProcessor(self.test_csv_path)
        processor.load_csv()
        
        # Testa colunas inexistentes
        if not processor.validate_columns('coluna_inexistente', 'outra_inexistente'):
            print("  ✓ Validação de colunas inexistentes funcionando")
        
        # Testa CSV vazio
        print("\n3. Testando CSV vazio...")
        empty_df = pd.DataFrame(columns=['col1', 'col2'])
        empty_df.to_csv('empty_test.csv', index=False)
        
        empty_processor = CSVProcessor('empty_test.csv')
        empty_processor.load_csv()
        empty_processor.validate_columns('col1', 'col2')
        empty_pairs = empty_processor.get_sentence_pairs()
        print(f"  ✓ CSV vazio: {len(empty_pairs)} pares (esperado: 0)")
        
        # Limpa arquivo temporário
        os.remove('empty_test.csv')
        
        return True
    
    def limpar_arquivos_teste(self):
        """Remove arquivos de teste"""
        print("\n=== Limpando Arquivos de Teste ===\n")
        
        arquivos = [self.test_csv_path, self.result_csv_path]
        for arquivo in arquivos:
            if Path(arquivo).exists():
                os.remove(arquivo)
                print(f"✓ Removido: {arquivo}")
    
    async def executar_teste_completo(self):
        """Executa todos os testes"""
        print("🧪 INICIANDO TESTE COMPLETO DO SISTEMA CSV\n")
        
        try:
            # Setup
            if not await self.setup():
                return False
            
            # Cria dataset
            if not self.criar_dataset_teste():
                return False
            
            # Testa carregamento
            processor, sentence_pairs = self.testar_carregamento_csv()
            if not processor:
                return False
            
            # Testa julgamento
            results = await self.testar_julgamento(sentence_pairs)
            if not results:
                return False
            
            # Testa salvamento
            if not self.testar_salvamento_resultados(processor, results):
                return False
            
            # Testa casos especiais
            if not self.testar_casos_especiais():
                return False
            
            print("\n🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
            return True
            
        except Exception as e:
            print(f"\n❌ ERRO DURANTE OS TESTES: {e}")
            logger.exception("Erro detalhado:")
            return False
        
        finally:
            # Limpa arquivos
            self.limpar_arquivos_teste()

async def main():
    """Função principal do teste"""
    teste = TesteCSV()
    sucesso = await teste.executar_teste_completo()
    
    if sucesso:
        print("\n✅ SISTEMA CSV FUNCIONANDO PERFEITAMENTE!")
        print("Você pode usar o sistema com confiança.")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM!")
        print("Verifique os logs acima para detalhes.")

if __name__ == "__main__":
    asyncio.run(main())
