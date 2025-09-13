"""
Script de migração para atualizar imports nos arquivos de teste
"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: Path):
    """Atualiza imports em um arquivo"""
    
    if not file_path.exists() or file_path.suffix != '.py':
        return
    
    print(f"Atualizando imports em: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Padrões de import para substituir
        replacements = [
            # Imports do main.py antigo
            (r'from main import', 'from src.judger import JudgerSystem\nfrom src.models import ModelConfig, SentencePair\nfrom src.clients import OllamaClient\nfrom src.processors import CSVProcessor'),
            (r'import main', 'from src import judger, models, clients, processors'),
            
            # Imports específicos
            (r'ModelConfig', 'ModelConfig'),
            (r'SentencePair', 'SentencePair'),
            (r'JudgmentResponse', 'JudgmentResponse'),
            (r'CSVProcessor', 'CSVProcessor'),
            (r'OllamaClient', 'OllamaClient'),
            (r'JudgerSystem', 'JudgerSystem'),
            (r'PromptTemplate', 'PromptTemplate'),
        ]
        
        # Aplicar substituições
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Adicionar imports necessários no início do arquivo
        if 'from src.' in content and 'import asyncio' not in content:
            content = 'import asyncio\n' + content
        
        # Salvar arquivo atualizado
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Arquivo atualizado: {file_path}")
        
    except Exception as e:
        print(f"❌ Erro ao atualizar {file_path}: {e}")


def migrate_test_files():
    """Migra todos os arquivos de teste"""
    
    tests_dir = Path("tests")
    
    if not tests_dir.exists():
        print("Pasta tests não encontrada!")
        return
    
    print("=== Migrando arquivos de teste ===\n")
    
    # Listar arquivos Python na pasta tests
    python_files = list(tests_dir.glob("*.py"))
    
    if not python_files:
        print("Nenhum arquivo Python encontrado na pasta tests")
        return
    
    for file_path in python_files:
        update_imports_in_file(file_path)
    
    print(f"\n✓ Migração concluída para {len(python_files)} arquivos")


def create_backup():
    """Cria backup do main.py original"""
    
    main_original = Path("main.py")
    main_backup = Path("main_original_backup.py")
    
    if main_original.exists() and not main_backup.exists():
        try:
            with open(main_original, 'r', encoding='utf-8') as f:
                content = f.read()
            
            with open(main_backup, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Backup criado: {main_backup}")
            
        except Exception as e:
            print(f"❌ Erro ao criar backup: {e}")


def main():
    """Função principal de migração"""
    
    print("=== Script de Migração LLM Judger ===\n")
    
    # Criar backup
    create_backup()
    
    # Migrar arquivos de teste
    migrate_test_files()
    
    print("\n=== Migração concluída ===")
    print("\nPróximos passos:")
    print("1. Teste os arquivos migrados na pasta tests/")
    print("2. Use main_new.py como novo arquivo principal")
    print("3. Configure o arquivo .env com suas preferências")
    print("4. Execute: python main_new.py")


if __name__ == "__main__":
    main()
