import asyncio
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import aiohttp
from dotenv import load_dotenv

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ModelConfig:
    name: str
    instances: int
    temperature: float = 0.1  # Baixo para consistência em julgamentos
    max_tokens: int = 512

@dataclass
class SentencePair:
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    context: Optional[str] = None
    reference_id: Optional[str] = None

@dataclass
class JudgmentResponse:
    model: str
    sentence_pair: SentencePair
    is_correct: bool
    confidence_score: float
    reasoning: str
    timestamp: datetime
    success: bool
    error: Optional[str] = None

class PromptTemplate:
    """Templates de prompts especializados para LLM Judger"""
    
    TRANSLATION_JUDGE_TEMPLATE = """Você é um especialista em avaliação de traduções. Sua tarefa é analisar se a tradução está correta, considerando precisão, fluência e adequação cultural.

## INSTRUÇÕES:
1. Avalie se a tradução captura o significado original
2. Considere a fluência no idioma de destino
3. Verifique se há erros gramaticais ou de contexto
4. Considere nuances culturais e idiomáticas
5. Retorne APENAS um JSON válido com sua avaliação

## FORMATO DE RESPOSTA OBRIGATÓRIO:
IMPORTANTE: Responda APENAS com um JSON válido, sem texto adicional, sem markdown, sem explicações.

{{
    "is_correct": true,
    "confidence_score": 0.85,
    "reasoning": "A tradução captura corretamente o significado original e mantém a fluência no idioma de destino"
}}

## DADOS PARA AVALIAÇÃO:

**Texto Original ({source_lang}):**
{source_text}

**Tradução ({target_lang}):**
{target_text}

{context_section}

**RESPOSTA (apenas JSON):**"""

    SEMANTIC_EQUIVALENCE_TEMPLATE = """Você é um especialista em análise semântica. Avalie se duas sentenças em idiomas diferentes expressam o mesmo significado, independentemente de serem traduções literais.

## INSTRUÇÕES:
1. Foque na equivalência semântica, não na tradução literal
2. Considere diferentes formas de expressar a mesma ideia
3. Avalie se a intenção comunicativa é preservada
4. Considere variações culturais legítimas
5. Retorne APENAS um JSON válido

## FORMATO DE RESPOSTA OBRIGATÓRIO:
IMPORTANTE: Responda APENAS com um JSON válido, sem texto adicional, sem markdown, sem explicações.

{{
    "is_correct": true,
    "confidence_score": 0.85,
    "reasoning": "As sentenças expressam o mesmo significado semântico, preservando a intenção comunicativa"
}}

## DADOS PARA AVALIAÇÃO:

**Sentença 1 ({source_lang}):**
{source_text}

**Sentença 2 ({target_lang}):**
{target_text}

{context_section}

**RESPOSTA (apenas JSON):**"""

    QUALITY_ASSESSMENT_TEMPLATE = """Você é um avaliador de qualidade linguística. Analise ambas as sentenças quanto à qualidade, clareza e adequação, independentemente da relação entre elas.

## CRITÉRIOS DE AVALIAÇÃO:
1. **Gramática e Sintaxe**: Correção linguística
2. **Clareza**: Facilidade de compreensão
3. **Naturalidade**: Fluência no idioma
4. **Adequação**: Apropriada para o contexto
5. **Completude**: Transmite informação completa

## FORMATO DE RESPOSTA OBRIGATÓRIO:
IMPORTANTE: Responda APENAS com um JSON válido, sem texto adicional, sem markdown, sem explicações.

{{
    "is_correct": true,
    "confidence_score": 0.85,
    "reasoning": "Ambas as sentenças apresentam boa qualidade linguística, com gramática correta e clareza adequada"
}}

## DADOS PARA AVALIAÇÃO:

**Texto 1 ({source_lang}):**
{source_text}

**Texto 2 ({target_lang}):**
{target_text}

{context_section}

**RESPOSTA (apenas JSON):**"""

    @classmethod
    def get_template(cls, template_type: str) -> str:
        """Retorna o template especificado"""
        templates = {
            "translation": cls.TRANSLATION_JUDGE_TEMPLATE,
            "semantic": cls.SEMANTIC_EQUIVALENCE_TEMPLATE,
            "quality": cls.QUALITY_ASSESSMENT_TEMPLATE
        }
        return templates.get(template_type, cls.TRANSLATION_JUDGE_TEMPLATE)

    @classmethod
    def format_prompt(cls, 
                     template_type: str, 
                     sentence_pair: SentencePair) -> str:
        """Formata o prompt com os dados da sentença"""
        
        template = cls.get_template(template_type)
        
        context_section = ""
        if sentence_pair.context:
            context_section = f"\n**Contexto Adicional:**\n{sentence_pair.context}\n"
        
        return template.format(
            source_text=sentence_pair.source_text,
            target_text=sentence_pair.target_text,
            source_lang=sentence_pair.source_language,
            target_lang=sentence_pair.target_language,
            context_section=context_section
        )

class OllamaClient:
    """Cliente para Ollama com controle de concorrência"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 max_concurrent_requests: int = 4,
                 timeout: int = 60):
        
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
    async def get_judgment(self, 
                          prompt: str, 
                          model: str,
                          temperature: float = 0.1,
                          max_retries: int = 3) -> Dict[str, Any]:
        """
        Faz chamada para Ollama com retry
        """
        
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession(timeout=self.timeout) as session:
                        payload = {
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "top_p": 0.9,
                                "top_k": 40,
                                "num_predict": 512
                            }
                        }
                        
                        async with session.post(
                            f"{self.base_url}/api/generate", 
                            json=payload
                        ) as response:
                            
                            if response.status == 200:
                                data = await response.json()
                                # Ollama retorna a resposta no campo "response"
                                content = data.get("response", "")
                                logger.debug(f"Resposta do Ollama: {content[:200]}...")
                                return {
                                    "success": True,
                                    "content": content,
                                    "model": model,
                                    "tokens": data.get("eval_count", 0),
                                    "prompt_eval_count": data.get("prompt_eval_count", 0)
                                }
                            else:
                                error_text = await response.text()
                                raise Exception(f"HTTP {response.status}: {error_text}")
                                
                except Exception as e:
                    logger.warning(f"Tentativa {attempt + 1} falhou para {model}: {e}")
                    logger.debug(f"Erro detalhado: {type(e).__name__}: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Aguardando {wait_time}s antes da próxima tentativa...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Todas as tentativas falharam para {model}")
                        return {
                            "success": False,
                            "content": "",
                            "model": model,
                            "error": str(e),
                            "tokens": 0
                        }

    async def test_connection(self) -> bool:
        """Testa se o Ollama está rodando e acessível"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        logger.info("✓ Conexão com Ollama estabelecida")
                        return True
                    else:
                        logger.error(f"❌ Ollama retornou status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Erro ao conectar com Ollama: {e}")
            return False

    async def test_model(self, model_name: str) -> bool:
        """Testa se um modelo específico está funcionando"""
        try:
            test_prompt = "Responda apenas: OK"
            result = await self.get_judgment(test_prompt, model_name, temperature=0.1)
            if result["success"] and result["content"].strip():
                logger.info(f"✓ Modelo {model_name} testado com sucesso")
                return True
            else:
                logger.warning(f"❌ Modelo {model_name} não respondeu corretamente")
                return False
        except Exception as e:
            logger.error(f"❌ Erro ao testar modelo {model_name}: {e}")
            return False

    async def list_models(self) -> List[str]:
        """Lista modelos disponíveis no Ollama"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        logger.info(f"Modelos encontrados: {models}")
                        return models
                    else:
                        logger.error(f"Erro ao listar modelos: HTTP {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {e}")
            return []

class JudgerSystem:
    """Sistema principal de julgamento com LLMs locais"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        
    def _parse_judgment_response(self, content: str) -> Dict[str, Any]:
        """Extrai e valida resposta JSON do modelo"""
        try:
            logger.debug(f"Tentando parsear resposta: {content[:300]}...")
            
            # Remove markdown code blocks se existirem
            content_clean = content.strip()
            if content_clean.startswith('```json'):
                content_clean = content_clean[7:]
            if content_clean.startswith('```'):
                content_clean = content_clean[3:]
            if content_clean.endswith('```'):
                content_clean = content_clean[:-3]
            
            # Tenta extrair JSON da resposta - procura por múltiplos padrões
            json_str = None
            
            # Padrão 1: JSON completo entre chaves
            start_idx = content_clean.find('{')
            end_idx = content_clean.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content_clean[start_idx:end_idx].strip()
            
            # Padrão 2: Se não encontrou JSON completo, tenta construir um
            if not json_str or not json_str.startswith('{'):
                # Tenta extrair campos individuais
                import re
                
                is_correct_match = re.search(r'"is_correct"\s*:\s*(true|false)', content_clean, re.IGNORECASE)
                confidence_match = re.search(r'"confidence_score"\s*:\s*(\d+\.?\d*)', content_clean)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content_clean)
                
                if is_correct_match and confidence_match and reasoning_match:
                    json_str = f'''{{
                        "is_correct": {is_correct_match.group(1).lower()},
                        "confidence_score": {confidence_match.group(1)},
                        "reasoning": "{reasoning_match.group(1)}"
                    }}'''
            
            if not json_str:
                raise ValueError("Nenhum JSON válido encontrado na resposta")
            
            logger.debug(f"JSON extraído: {json_str}")
            
            parsed = json.loads(json_str)
            
            # Valida campos obrigatórios
            required_fields = ["is_correct", "confidence_score", "reasoning"]
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Campo obrigatório '{field}' não encontrado")
            
            # Valida tipos
            if not isinstance(parsed["is_correct"], bool):
                parsed["is_correct"] = bool(parsed["is_correct"])
            
            if not isinstance(parsed["confidence_score"], (int, float)):
                parsed["confidence_score"] = 0.5
            
            # Garante que confidence_score está entre 0 e 1
            parsed["confidence_score"] = max(0.0, min(1.0, float(parsed["confidence_score"])))
            
            logger.debug(f"JSON parseado com sucesso: {parsed}")
            return parsed
            
        except Exception as e:
            logger.warning(f"Erro ao parsear resposta JSON: {e}")
            logger.debug(f"Conteúdo original: {content}")
            
            # Fallback: tenta extrair informação básica
            content_lower = content.lower()
            is_correct = any(word in content_lower 
                           for word in ["correct", "true", "correto", "verdadeiro", "sim", "yes"])
            
            # Tenta extrair confidence score
            confidence = 0.5
            if "confidence" in content_lower:
                import re
                conf_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', content_lower)
                if conf_match:
                    try:
                        confidence = float(conf_match.group(1))
                        if confidence > 1:
                            confidence = confidence / 100  # Assume que está em porcentagem
                    except:
                        pass
            
            return {
                "is_correct": is_correct,
                "confidence_score": confidence,
                "reasoning": f"Resposta não estruturada: {content[:200]}...",
                "parsing_error": str(e)
            }
    
    async def judge_sentence_pair(self, 
                                sentence_pair: SentencePair,
                                model: str,
                                template_type: str = "translation") -> JudgmentResponse:
        """Julga um par de sentenças"""
        
        prompt = PromptTemplate.format_prompt(template_type, sentence_pair)
        
        result = await self.ollama_client.get_judgment(
            prompt=prompt,
            model=model,
            temperature=0.1  # Baixo para consistência
        )
        
        if result["success"]:
            try:
                judgment = self._parse_judgment_response(result["content"])
                
                return JudgmentResponse(
                    model=model,
                    sentence_pair=sentence_pair,
                    is_correct=judgment["is_correct"],
                    confidence_score=float(judgment["confidence_score"]),
                    reasoning=judgment["reasoning"],
                    timestamp=datetime.now(),
                    success=True
                )
                
            except Exception as e:
                logger.error(f"Erro ao processar julgamento: {e}")
                return JudgmentResponse(
                    model=model,
                    sentence_pair=sentence_pair,
                    is_correct=False,
                    confidence_score=0.0,
                    reasoning=f"Erro no processamento: {e}",
                    timestamp=datetime.now(),
                    success=False,
                    error=str(e)
                )
        else:
            return JudgmentResponse(
                model=model,
                sentence_pair=sentence_pair,
                is_correct=False,
                confidence_score=0.0,
                reasoning="Falha na comunicação com o modelo",
                timestamp=datetime.now(),
                success=False,
                error=result.get("error", "Erro desconhecido")
            )
    
    async def batch_judgment(self, 
                           sentence_pairs: List[SentencePair],
                           configs: List[ModelConfig],
                           template_type: str = "translation") -> List[JudgmentResponse]:
        """Processa múltiplos pares com múltiplos modelos"""
        
        tasks = []
        for config in configs:
            for sentence_pair in sentence_pairs:
                for _ in range(config.instances):
                    task = asyncio.create_task(
                        self.judge_sentence_pair(sentence_pair, config.name, template_type)
                    )
                    tasks.append(task)
        
        logger.info(f"Iniciando {len(tasks)} julgamentos")
        
        results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1
                
                status = "✓" if result.success else "✗"
                logger.info(f"{status} [{completed}/{len(tasks)}] {result.model}: {result.confidence_score:.2f}")
                
            except Exception as e:
                logger.error(f"Erro não tratado no julgamento: {e}")
                logger.debug(f"Tipo do erro: {type(e).__name__}")
                # Cria um resultado de erro para manter a contagem
                error_result = JudgmentResponse(
                    model="unknown",
                    sentence_pair=SentencePair("", "", "", ""),
                    is_correct=False,
                    confidence_score=0.0,
                    reasoning=f"Erro não tratado: {e}",
                    timestamp=datetime.now(),
                    success=False,
                    error=str(e)
                )
                results.append(error_result)
                completed += 1
        
        return results

async def main():
    """Função principal do sistema judger"""
    
    print("=== Sistema LLM Judger com Ollama ===\n")
    
    # Opção para debug
    debug_mode = input("Ativar modo debug? (y/n): ").strip().lower() == 'y'
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        print("✓ Modo debug ativado")
    
    # Inicializa cliente Ollama
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    client = OllamaClient(base_url=ollama_url, max_concurrent_requests=4)
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
    template_types = {
        "1": ("translation", "Avaliação de Tradução"),
        "2": ("semantic", "Equivalência Semântica"), 
        "3": ("quality", "Avaliação de Qualidade")
    }
    
    print(f"\nTipos de avaliação:")
    for key, (_, desc) in template_types.items():
        print(f"{key}: {desc}")
    
    template_choice = input("Escolha o tipo de avaliação (1-3): ").strip()
    template_type, template_desc = template_types.get(template_choice, ("translation", "Avaliação de Tradução"))
    
    print(f"\nUsando: {template_desc}")
    print("\nConfiguração ativa:")
    for config in configs:
        print(f"- {config.instances}x {config.name}")
    
    # Loop principal de julgamento
    sentence_pairs = []
    
    while True:
        print(f"\n=== Entrada de Dados ===")
        print("Digite os pares de sentenças para avaliar (ou 'process' para processar, 'exit' para sair):")
        
        command = input("\nComando: ").strip().lower()
        
        if command == 'exit':
            break
        elif command == 'process':
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
            
        else:
            # Coleta dados do par de sentenças
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

if __name__ == "__main__":
    asyncio.run(main())