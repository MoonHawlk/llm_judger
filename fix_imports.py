#!/usr/bin/env python3
"""
Script para corrigir imports em todos os arquivos de teste
"""

import os
from pathlib import Path


def fix_imports_in_file(file_path: Path):
    """Corrige imports em um arquivo de teste"""
    
    if not file_path.exists() or file_path.suffix != '.py' or file_path.name == '__init__.py':
        return
    
    print(f"Corrigindo imports em: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar se já tem o path fix
        if 'sys.path.insert(0, str(parent_dir))' in content:
            print(f"  ✓ {file_path.name} já está corrigido")
            return
        
        # Adicionar imports de path no início
        path_imports = '''import sys
import os
from pathlib import Path

# Adicionar o diretório pai ao path para importar src
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

'''
        
        # Encontrar onde inserir os imports
        lines = content.split('\n')
        insert_index = 0
        
        # Pular shebang e docstring
        for i, line in enumerate(lines):
            if line.startswith('#!/'):
                insert_index = i + 1
            elif line.startswith('"""') or line.startswith("'''"):
                # Encontrar o fim da docstring
                for j in range(i + 1, len(lines)):
                    if lines[j].endswith('"""') or lines[j].endswith("'''"):
                        insert_index = j + 1
                        break
                break
            elif line.strip() and not line.startswith('#'):
                insert_index = i
                break
        
        # Inserir os imports
        lines.insert(insert_index, path_imports.strip())
        
        # Salvar arquivo corrigido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"  ✓ {file_path.name} corrigido")
        
    except Exception as e:
        print(f"  ❌ Erro ao corrigir {file_path.name}: {e}")


def main():
    """Função principal"""
    
    print("=== Corrigindo imports nos arquivos de teste ===\n")
    
    tests_dir = Path("tests")
    
    if not tests_dir.exists():
        print("Pasta tests não encontrada!")
        return
    
    # Listar arquivos Python na pasta tests
    python_files = list(tests_dir.glob("*.py"))
    
    if not python_files:
        print("Nenhum arquivo Python encontrado na pasta tests")
        return
    
    for file_path in python_files:
        fix_imports_in_file(file_path)
    
    print(f"\n✓ Correção concluída para {len(python_files)} arquivos")


if __name__ == "__main__":
    main()
