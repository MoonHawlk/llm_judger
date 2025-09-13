"""
Clientes para comunicação com LLMs
"""

import asyncio
import logging
from typing import List, Dict, Any
import aiohttp
import json

logger = logging.getLogger(__name__)


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
