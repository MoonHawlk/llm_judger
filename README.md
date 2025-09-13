# LLM Judger com Ollama

Sistema para avaliação de traduções e equivalência semântica usando modelos de linguagem locais via Ollama.

## 🚀 Características

- **Integração com Ollama**: Usa modelos LLM locais para avaliação
- **Múltiplos tipos de avaliação**: Tradução, equivalência semântica, qualidade
- **Processamento em lote**: Avalia múltiplos pares de sentenças simultaneamente
- **Processamento de datasets CSV**: Carrega e processa arquivos CSV automaticamente
- **Concorrência controlada**: Gerencia múltiplas instâncias de modelos
- **Logging detalhado**: Debug e monitoramento de operações
- **Tratamento robusto de erros**: Retry automático e fallbacks
- **Exportação de resultados**: Salva resultados diretamente no CSV com colunas adicionais

## 📋 Pré-requisitos

1. **Ollama instalado e rodando**:
   ```bash
   # Instalar Ollama (se não estiver instalado)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Iniciar o servidor Ollama
   ollama serve
   ```

2. **Modelos baixados**:
   ```bash
   # Exemplos de modelos recomendados
   ollama pull llama2
   ollama pull mistral
   ollama pull codellama
   ```

3. **Dependências Python**:
   ```bash
   pip install aiohttp python-dotenv pandas
   ```

## 🛠️ Instalação

1. Clone ou baixe o projeto
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Certifique-se de que o Ollama está rodando:
   ```bash
   ollama serve
   ```

## 🎯 Como Usar

### Uso Interativo

Execute o script principal:

```bash
python main.py
```

O sistema irá:
1. Testar a conexão com o Ollama
2. Listar modelos disponíveis
3. Permitir configurar modelos e instâncias
4. Oferecer opções de entrada:
   - Adicionar pares manualmente
   - Processar dataset CSV
   - Processar pares adicionados
5. Processar e mostrar resultados

### Uso Programático

Veja o arquivo `exemplo_uso.py` para exemplos de como usar o sistema programaticamente:

```python
from main import OllamaClient, JudgerSystem, SentencePair, ModelConfig

# Inicializar cliente
client = OllamaClient(base_url="http://localhost:11434")
judger = JudgerSystem(client)

# Criar par de sentenças
pair = SentencePair(
    source_text="Hello, how are you?",
    target_text="Olá, como você está?",
    source_language="en",
    target_language="pt"
)

# Fazer julgamento
result = await judger.judge_sentence_pair(pair, "llama2", "translation")
```

### Processamento de Datasets CSV

O sistema suporta processamento automático de datasets CSV:

```python
from main import CSVProcessor, JudgerSystem, ModelConfig

# Carregar dataset CSV
processor = CSVProcessor("meu_dataset.csv")
processor.load_csv()

# Configurar colunas e idiomas
processor.validate_columns("texto_original", "texto_traduzido")
processor.set_languages("en", "pt")

# Converter para pares de sentenças
sentence_pairs = processor.get_sentence_pairs()

# Processar com modelo
configs = [ModelConfig(name="llama2", instances=2)]
results = await judger.batch_judgment(sentence_pairs, configs, "translation")

# Salvar resultados no CSV
processor.save_results(results, "resultados.csv")
```

#### Formato do CSV

O CSV deve ter pelo menos duas colunas com os textos a serem avaliados:

```csv
texto_original,texto_traduzido
Hello, how are you?,Olá, como você está?
Good morning!,Bom dia!
Thank you very much.,Muito obrigado.
```

#### Colunas de Resultado

Após o processamento, o CSV será expandido com as seguintes colunas:

- **resultado**: "Correto" ou "Incorreto"
- **explicacao**: Explicação detalhada do julgamento
- **confianca**: Score de confiança (0.0-1.0)
- **modelo**: Nome do modelo usado
- **timestamp**: Data e hora do processamento

## 📊 Tipos de Avaliação

### 1. Avaliação de Tradução
- Verifica precisão da tradução
- Considera fluência e adequação cultural
- Identifica erros gramaticais e de contexto

### 2. Equivalência Semântica
- Foca no significado, não na tradução literal
- Considera diferentes formas de expressar a mesma ideia
- Avalia preservação da intenção comunicativa

### 3. Avaliação de Qualidade
- Analisa qualidade linguística de ambas as sentenças
- Considera gramática, clareza, naturalidade
- Avalia adequação ao contexto

## ⚙️ Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` (opcional):

```env
OLLAMA_URL=http://localhost:11434
```

### Parâmetros do Cliente

```python
client = OllamaClient(
    base_url="http://localhost:11434",  # URL do Ollama
    max_concurrent_requests=4,          # Máximo de requisições simultâneas
    timeout=60                          # Timeout em segundos
)
```

## 🔧 Solução de Problemas

### Ollama não conecta
- Verifique se o Ollama está rodando: `ollama serve`
- Confirme a URL: padrão é `http://localhost:11434`
- Verifique firewall/proxy

### Nenhum modelo encontrado
- Liste modelos: `ollama list`
- Baixe um modelo: `ollama pull llama2`

### Modelo não responde
- Teste o modelo: `ollama run llama2`
- Verifique se há memória suficiente
- Tente um modelo menor

### Erro de parsing JSON
- Ative modo debug para ver respostas detalhadas
- O sistema tem fallback para respostas não estruturadas

## 📈 Monitoramento

O sistema inclui logging detalhado:

- **INFO**: Operações principais e status
- **WARNING**: Problemas recuperáveis
- **ERROR**: Falhas críticas
- **DEBUG**: Detalhes técnicos (ativar com modo debug)

## 🤝 Contribuição

Para contribuir:
1. Faça fork do projeto
2. Crie uma branch para sua feature
3. Implemente e teste
4. Submeta um pull request

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

## 📁 Arquivos do Projeto

- **`main.py`**: Sistema principal com todas as funcionalidades
- **`exemplo_uso.py`**: Exemplos de uso programático
- **`exemplo_csv.py`**: Exemplo de processamento de CSV
- **`teste_rapido.py`**: Script de teste rápido
- **`exemplo_dataset.csv`**: Dataset de exemplo para testes
- **`requirements.txt`**: Dependências do projeto
- **`README.md`**: Esta documentação

## 🆘 Suporte

Para problemas ou dúvidas:
1. Verifique a seção de solução de problemas
2. Ative o modo debug para mais informações
3. Teste com o arquivo `exemplo_dataset.csv` fornecido
4. Execute `python teste_rapido.py` para verificar a instalação
5. Abra uma issue no repositório