# LLM Judger - Sistema Modularizado

Sistema de julgamento com LLMs locais usando Ollama, agora com arquitetura modular e organizada.

## 📁 Estrutura do Projeto

```
llm_judger/
├── src/                    # Código fonte principal
│   ├── __init__.py
│   ├── models.py          # Modelos de dados (dataclasses)
│   ├── clients.py         # Cliente Ollama
│   ├── processors.py      # Processamento de CSV
│   ├── templates.py       # Templates de prompts
│   ├── judger.py          # Sistema principal de julgamento
│   └── utils.py           # Utilitários e funções auxiliares
├── config/                # Configurações
│   ├── __init__.py
│   └── settings.py        # Configurações centralizadas
├── data/                  # Arquivos CSV de dados
│   └── exemplo_dataset.csv
├── tests/                 # Arquivos de teste e demonstração
│   ├── __init__.py
│   ├── demo_csv.py
│   ├── exemplo_csv.py
│   ├── exemplo_uso.py
│   ├── teste_csv.py
│   ├── teste_csv_simples.py
│   └── teste_mapeamento.py
├── main_new.py           # Arquivo principal modularizado
├── main.py               # Arquivo principal original (backup)
├── requirements.txt      # Dependências
├── .env                  # Variáveis de ambiente
└── README.md            # Documentação original
```

## 🚀 Como Usar

### 1. Instalação

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar Ollama (se ainda não estiver instalado)
# https://ollama.ai/
```

### 2. Configuração

Crie um arquivo `.env` na raiz do projeto:

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

# Configurações de retry
MAX_RETRIES=3
RETRY_DELAY_BASE=2
```

### 3. Execução

```bash
# Executar o sistema modularizado
python main_new.py

# Ou executar o sistema original (backup)
python main.py
```

## 📋 Funcionalidades

### Tipos de Avaliação

1. **Avaliação de Tradução**: Analisa se uma tradução está correta
2. **Equivalência Semântica**: Verifica se duas sentenças expressam o mesmo significado
3. **Avaliação de Qualidade**: Avalia a qualidade linguística das sentenças

### Processamento de Dados

- **Entrada Manual**: Adicione pares de sentenças manualmente
- **Processamento CSV**: Carregue datasets em formato CSV
- **Múltiplos Modelos**: Use vários modelos simultaneamente
- **Processamento em Lote**: Processe múltiplos pares de uma vez

### Recursos Avançados

- **Controle de Concorrência**: Limite o número de requisições simultâneas
- **Sistema de Retry**: Tentativas automáticas em caso de falha
- **Logging Detalhado**: Registros completos de todas as operações
- **Análise de Resultados**: Estatísticas detalhadas dos julgamentos

## 🔧 Módulos Principais

### `src/models.py`
Contém as classes de dados principais:
- `ModelConfig`: Configuração de modelos
- `SentencePair`: Par de sentenças para julgamento
- `JudgmentResponse`: Resposta de julgamento
- `CSVRow`: Linha de dados CSV

### `src/clients.py`
Cliente para comunicação com Ollama:
- Conexão e autenticação
- Listagem de modelos disponíveis
- Teste de modelos
- Controle de concorrência

### `src/processors.py`
Processamento de dados CSV:
- Carregamento e validação de CSVs
- Extração de pares de sentenças
- Salvamento de resultados
- Suporte a múltiplos encodings

### `src/templates.py`
Templates de prompts especializados:
- Templates para diferentes tipos de avaliação
- Formatação automática de prompts
- Suporte a contexto adicional

### `src/judger.py`
Sistema principal de julgamento:
- Processamento de julgamentos individuais
- Processamento em lote
- Parsing de respostas JSON
- Tratamento de erros

### `src/utils.py`
Utilitários e funções auxiliares:
- Configuração de logging
- Análise de resultados
- Formatação de dados
- Funções de estatística

### `config/settings.py`
Configurações centralizadas:
- Variáveis de ambiente
- Configurações padrão
- Criação de diretórios
- Constantes do sistema

## 📊 Exemplo de Uso

```python
from src.models import ModelConfig, SentencePair
from src.clients import OllamaClient
from src.judger import JudgerSystem

# Configurar cliente
client = OllamaClient()
judger = JudgerSystem(client)

# Configurar modelo
config = ModelConfig(name="llama2", instances=2)

# Criar par de sentenças
pair = SentencePair(
    source_text="Hello world",
    target_text="Olá mundo",
    source_language="en",
    target_language="pt"
)

# Fazer julgamento
result = await judger.judge_sentence_pair(pair, "llama2", "translation")
print(f"Correto: {result.is_correct}")
print(f"Confiança: {result.confidence_score}")
```

## 🧪 Testes

Os arquivos de teste estão na pasta `tests/`:

- `demo_csv.py`: Demonstração de processamento CSV
- `exemplo_csv.py`: Exemplo de uso com CSV
- `teste_csv.py`: Testes de processamento CSV
- `teste_rapido.py`: Testes rápidos do sistema

## 📈 Melhorias Implementadas

1. **Modularização**: Código separado em módulos lógicos
2. **Configuração Centralizada**: Todas as configurações em um local
3. **Organização de Arquivos**: Estrutura clara de pastas
4. **Reutilização**: Módulos podem ser importados independentemente
5. **Manutenibilidade**: Código mais fácil de manter e estender
6. **Testabilidade**: Módulos podem ser testados individualmente

## 🔄 Migração

Para migrar do sistema antigo para o novo:

1. Use `main_new.py` em vez de `main.py`
2. Atualize imports se estiver usando o código como biblioteca
3. Configure o arquivo `.env` com suas preferências
4. Os arquivos CSV devem estar na pasta `data/`

## 📝 Logs

O sistema gera logs detalhados incluindo:
- Conexões com Ollama
- Processamento de julgamentos
- Erros e tentativas de retry
- Estatísticas de performance
- Resultados de análise

## 🤝 Contribuição

Para contribuir com o projeto:

1. Mantenha a estrutura modular
2. Adicione testes para novas funcionalidades
3. Documente mudanças no README
4. Siga as convenções de nomenclatura Python
