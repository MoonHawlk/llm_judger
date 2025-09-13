# Guia de Migração - LLM Judger

## ✅ Organização Concluída

Seu projeto foi completamente reorganizado e modularizado! Aqui está o que foi feito:

## 📁 Nova Estrutura

```
llm_judger/
├── src/                    # Código fonte modularizado
│   ├── models.py          # Modelos de dados
│   ├── clients.py         # Cliente Ollama
│   ├── processors.py      # Processamento CSV
│   ├── templates.py       # Templates de prompts
│   ├── judger.py          # Sistema de julgamento
│   └── utils.py           # Utilitários
├── config/                # Configurações centralizadas
│   └── settings.py        # Todas as configurações
├── data/                  # Arquivos CSV organizados
│   └── exemplo_dataset.csv
├── tests/                 # Testes organizados
│   ├── demo_csv.py
│   ├── exemplo_csv.py
│   └── [outros testes...]
├── main_new.py           # Novo arquivo principal
├── main.py               # Arquivo original (mantido)
└── exemplo_modular.py    # Exemplo de uso
```

## 🚀 Como Usar o Sistema Reorganizado

### 1. Executar o Sistema Principal

```bash
# Use o novo arquivo principal modularizado
python main_new.py
```

### 2. Executar Exemplos

```bash
# Exemplo completo de uso
python exemplo_modular.py
```

### 3. Executar Testes

```bash
# Testes individuais
python tests/demo_csv.py
python tests/exemplo_csv.py
```

## 🔧 Configuração

### Arquivo .env

Crie ou atualize o arquivo `.env`:

```env
# Configurações do Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MAX_CONCURRENT_REQUESTS=4
OLLAMA_TIMEOUT=60

# Configurações de logging
LOG_LEVEL=INFO
DEBUG_MODE=false

# Configurações de modelo
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=512
```

## 📋 Principais Melhorias

### ✅ Organização
- **Arquivos CSV** movidos para `data/`
- **Testes** organizados em `tests/`
- **Código fonte** modularizado em `src/`
- **Configurações** centralizadas em `config/`

### ✅ Modularização
- `models.py`: Classes de dados (ModelConfig, SentencePair, etc.)
- `clients.py`: Cliente Ollama com controle de concorrência
- `processors.py`: Processamento de arquivos CSV
- `templates.py`: Templates de prompts especializados
- `judger.py`: Sistema principal de julgamento
- `utils.py`: Funções auxiliares e utilitários

### ✅ Configuração Centralizada
- Todas as configurações em `config/settings.py`
- Suporte a variáveis de ambiente
- Configurações padrão sensatas

### ✅ Reutilização
- Módulos podem ser importados independentemente
- Código mais fácil de testar e manter
- Estrutura escalável para futuras funcionalidades

## 🔄 Migração dos Arquivos

### Arquivos Atualizados
- ✅ Todos os arquivos de teste foram migrados
- ✅ Imports atualizados automaticamente
- ✅ Backup do `main.py` original criado

### Arquivos Mantidos
- `main.py`: Arquivo original (para referência)
- `main_old.py`: Versão anterior
- `main_original_backup.py`: Backup criado automaticamente

## 🧪 Testando a Nova Estrutura

### 1. Teste Básico
```bash
python exemplo_modular.py
```

### 2. Teste com CSV
```bash
python tests/demo_csv.py
```

### 3. Teste Completo
```bash
python main_new.py
```

## 📚 Documentação

- `README_NEW.md`: Documentação completa da nova estrutura
- `MIGRATION_GUIDE.md`: Este guia de migração
- `exemplo_modular.py`: Exemplos práticos de uso

## 🎯 Próximos Passos

1. **Teste o sistema**: Execute `python main_new.py`
2. **Configure o .env**: Ajuste as configurações conforme necessário
3. **Explore os exemplos**: Execute `python exemplo_modular.py`
4. **Use os testes**: Execute arquivos na pasta `tests/`

## 🔧 Desenvolvimento Futuro

### Adicionando Novos Módulos
```python
# Em src/novo_modulo.py
from .models import SentencePair
from .utils import setup_logging

# Sua implementação aqui
```

### Adicionando Novos Templates
```python
# Em src/templates.py
NOVO_TEMPLATE = """
Seu template aqui...
"""

# Adicionar ao método get_template()
```

### Adicionando Novos Processadores
```python
# Em src/processors.py
class NovoProcessor:
    def processar(self, dados):
        # Sua implementação
        pass
```

## 🆘 Solução de Problemas

### Erro de Import
```bash
# Se houver erro de import, verifique se está na pasta correta
cd llm_judger
python main_new.py
```

### Erro de Configuração
```bash
# Verifique o arquivo .env
cat .env
```

### Erro de Dependências
```bash
# Reinstale as dependências
pip install -r requirements.txt
```

## ✨ Benefícios da Nova Estrutura

1. **Manutenibilidade**: Código mais fácil de manter
2. **Escalabilidade**: Fácil adicionar novas funcionalidades
3. **Testabilidade**: Módulos podem ser testados individualmente
4. **Reutilização**: Código pode ser reutilizado em outros projetos
5. **Organização**: Estrutura clara e profissional
6. **Configuração**: Configurações centralizadas e flexíveis

---

**🎉 Parabéns! Seu projeto está agora completamente organizado e modularizado!**
