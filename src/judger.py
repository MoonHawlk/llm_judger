"""
Sistema principal de julgamento com LLMs
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any
from datetime import datetime

from .models import SentencePair, JudgmentResponse, ModelConfig
from .templates import PromptTemplate
from .clients import OllamaClient

logger = logging.getLogger(__name__)


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
