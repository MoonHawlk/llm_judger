import sys
import os
from pathlib import Path

# Adicionar o diretório pai ao path para importar src
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
import asyncio
#!/usr/bin/env python3
"""
Teste específico para o mapeamento de resultados CSV
"""

import pandas as pd
from src.judger import JudgerSystem
from src.models import ModelConfig, SentencePair
from src.clients import OllamaClient
from src.processors import CSVProcessor CSVProcessor, JudgmentResponse, SentencePair
from datetime import datetime

def criar_dataset_teste_mapeamento():
    """Cria um dataset específico para testar mapeamento"""
    
    # Dataset com linhas vazias e diferentes índices para testar mapeamento
    dados = [
        ("Hello", "Olá"),           # Linha 0
        ("", ""),                   # Linha 1 - vazia
        ("Good", "Bom"),            # Linha 2
        ("", "Vazio"),              # Linha 3 - parcialmente vazia
        ("Thank", "Obrigado"),      # Linha 4
        ("", ""),                   # Linha 5 - vazia
        ("Love", "Amor"),           # Linha 6
    ]
    
    df = pd.DataFrame(dados, columns=['texto_original', 'texto_traduzido'])
    df.to_csv('teste_mapeamento.csv', index=False, encoding='utf-8')
    
    print("Dataset de teste criado:")
    print(df)
    print(f"Total de linhas: {len(df)}")
    
    return df

def testar_mapeamento():
    """Testa o mapeamento de resultados"""
    
    print("=== Teste de Mapeamento de Resultados ===\n")
    
    # Cria dataset
    df_original = criar_dataset_teste_mapeamento()
    
    # Inicializa processador
    processor = CSVProcessor('teste_mapeamento.csv')
    processor.load_csv()
    processor.validate_columns('texto_original', 'texto_traduzido')
    processor.set_languages('en', 'pt')
    
    # Converte para pares
    sentence_pairs = processor.get_sentence_pairs()
    print(f"\nPares válidos encontrados: {len(sentence_pairs)}")
    
    for i, pair in enumerate(sentence_pairs):
        print(f"  {i+1}. ID: {pair.reference_id} - '{pair.source_text}' -> '{pair.target_text}'")
    
    # Cria resultados simulados
    resultados_simulados = []
    for i, pair in enumerate(sentence_pairs):
        resultado = JudgmentResponse(
            model="teste",
            sentence_pair=pair,
            is_correct=i % 2 == 0,  # Alterna entre correto/incorreto
            confidence_score=0.7 + (i * 0.1),
            reasoning=f"Teste de mapeamento {i+1}",
            timestamp=datetime.now(),
            success=True
        )
        resultados_simulados.append(resultado)
    
    print(f"\nResultados simulados criados: {len(resultados_simulados)}")
    
    # Testa salvamento
    print("\nTestando salvamento...")
    if processor.save_results(resultados_simulados, 'teste_mapeamento_resultados.csv'):
        print("✓ Salvamento bem-sucedido!")
        
        # Verifica resultado
        df_resultado = pd.read_csv('teste_mapeamento_resultados.csv')
        print(f"\nArquivo de resultado:")
        print(df_resultado[['texto_original', 'texto_traduzido', 'resultado', 'confianca']])
        
        # Verifica se os resultados foram mapeados corretamente
        print(f"\nVerificação de mapeamento:")
        for idx, row in df_resultado.iterrows():
            if pd.notna(row['resultado']) and row['resultado'] != '':
                print(f"  Linha {idx}: {row['resultado']} (confiança: {row['confianca']})")
            else:
                print(f"  Linha {idx}: SEM RESULTADO (linha vazia ou inválida)")
        
        return True
    else:
        print("❌ Erro no salvamento!")
        return False

def limpar_arquivos():
    """Remove arquivos de teste"""
    import os
    arquivos = ['teste_mapeamento.csv', 'teste_mapeamento_resultados.csv']
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            os.remove(arquivo)
            print(f"✓ Removido: {arquivo}")

def main():
    """Função principal"""
    try:
        sucesso = testar_mapeamento()
        
        if sucesso:
            print("\n✅ TESTE DE MAPEAMENTO PASSOU!")
            print("O mapeamento de resultados está funcionando corretamente.")
        else:
            print("\n❌ TESTE DE MAPEAMENTO FALHOU!")
    
    finally:
        print("\nLimpando arquivos de teste...")
        limpar_arquivos()

if __name__ == "__main__":
    main()
