import asyncio
import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ModelConfig:
    name: str
    instances: int
    max_tokens: int = 100

@dataclass
class APIResponse:
    model: str
    content: str
    timestamp: datetime
    tokens_used: int
    success: bool
    error: Optional[str] = None

class RateLimiter:
    """Controla rate limiting para evitar exceder limites da API"""
    
    def __init__(self, max_requests_per_minute: int = 50):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = datetime.now()
            # Remove requests mais antigos que 1 minuto
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < timedelta(minutes=1)]
            
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - min(self.requests)).total_seconds()
                if sleep_time > 0:
                    logger.info(f"Rate limit atingido. Aguardando {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self.requests.append(now)

class ScalableOpenAIClient:
    """Cliente escalável para OpenAI com controle de concorrência e rate limiting"""
    
    def __init__(self, 
                 api_key: str,
                 max_concurrent_requests: int = 10,
                 max_requests_per_minute: int = 50,
                 timeout: int = 30):
        
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        
    async def get_completion(self, 
                           prompt: str, 
                           model: str, 
                           max_tokens: int = 100,
                           max_retries: int = 3) -> APIResponse:
        """
        Faz chamada para OpenAI com controle de concorrência, rate limiting e retry
        """
        
        async with self.semaphore:  # Controla concorrência
            await self.rate_limiter.acquire()  # Controla rate limiting
            
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens
                    )
                    
                    content = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens
                    
                    return APIResponse(
                        model=model,
                        content=content,
                        timestamp=datetime.now(),
                        tokens_used=tokens_used,
                        success=True
                    )
                    
                except Exception as e:
                    logger.warning(f"Tentativa {attempt + 1} falhou para {model}: {e}")
                    
                    if attempt < max_retries - 1:
                        # Backoff exponencial
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    else:
                        return APIResponse(
                            model=model,
                            content="",
                            timestamp=datetime.now(),
                            tokens_used=0,
                            success=False,
                            error=str(e)
                        )
    
    async def process_batch(self, 
                          prompt: str, 
                          configs: List[ModelConfig]) -> List[APIResponse]:
        """Processa lote de requests de forma otimizada"""
        
        tasks = []
        for config in configs:
            for _ in range(config.instances):
                task = asyncio.create_task(
                    self.get_completion(prompt, config.name, config.max_tokens)
                )
                tasks.append(task)
        
        logger.info(f"Iniciando {len(tasks)} chamadas assíncronas")
        
        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                
                status = "✓" if result.success else "✗"
                logger.info(f"{status} {result.model}: {result.tokens_used} tokens")
                
            except Exception as e:
                logger.error(f"Erro não tratado: {e}")
        
        return results
    
    async def close(self):
        """Fecha conexões do cliente"""
        await self.client.close()

class ConfigManager:
    """Gerencia configurações de modelos de forma mais escalável"""
    
    AVAILABLE_MODELS = {
        "1": "gpt-3.5-turbo",
        "2": "gpt-4",
        "3": "gpt-4-turbo-preview"
    }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> List[ModelConfig]:
        """Cria configuração a partir de dicionário (útil para APIs/configs)"""
        configs = []
        for model_name, instances in config_dict.items():
            if model_name in cls.AVAILABLE_MODELS.values():
                configs.append(ModelConfig(name=model_name, instances=instances))
        return configs
    
    @classmethod
    def interactive_setup(cls) -> List[ModelConfig]:
        """Configuração interativa melhorada"""
        configs = []
        
        print("\nModelos disponíveis:")
        for key, model in cls.AVAILABLE_MODELS.items():
            print(f"{key}: {model}")
        
        while True:
            try:
                choice = input("\nEscolha o modelo (1-3) ou 'done': ").strip()
                
                if choice.lower() == 'done':
                    break
                    
                if choice not in cls.AVAILABLE_MODELS:
                    print("Opção inválida!")
                    continue
                
                instances = int(input("Número de instâncias: "))
                if instances < 1:
                    print("Deve ser pelo menos 1 instância!")
                    continue
                
                max_tokens = int(input("Max tokens (padrão 100): ") or "100")
                
                model_name = cls.AVAILABLE_MODELS[choice]
                configs.append(ModelConfig(
                    name=model_name, 
                    instances=instances,
                    max_tokens=max_tokens
                ))
                
                print(f"Adicionado: {instances}x {model_name}")
                
            except ValueError:
                print("Entrada inválida!")
                
        return configs

async def main():
    """Função principal escalável"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY não encontrada!")
        return
    
    # Cliente escalável com configurações otimizadas
    client = ScalableOpenAIClient(
        api_key=api_key,
        max_concurrent_requests=5,  # Ajuste conforme seus limites
        max_requests_per_minute=50  # Ajuste conforme sua conta
    )
    
    try:
        print("=== Sistema Escalável OpenAI ===")
        
        # Configuração (pode vir de arquivo, API, etc.)
        configs = ConfigManager.interactive_setup()
        
        if not configs:
            print("Nenhuma configuração definida!")
            return
        
        print(f"\nConfiguração ativa:")
        for config in configs:
            print(f"- {config.instances}x {config.name} ({config.max_tokens} tokens)")
        
        while True:
            prompt = input("\nDigite sua pergunta (ou 'exit'): ").strip()
            
            if prompt.lower() == 'exit':
                break
            
            if not prompt:
                continue
            
            # Processa lote de forma escalável
            start_time = datetime.now()
            results = await client.process_batch(prompt, configs)
            end_time = datetime.now()
            
            # Relatório de resultados
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            total_tokens = sum(r.tokens_used for r in successful)
            
            print(f"\n=== Resultados ===")
            print(f"Tempo total: {(end_time - start_time).total_seconds():.2f}s")
            print(f"Sucessos: {len(successful)}/{len(results)}")
            print(f"Tokens usados: {total_tokens}")
            
            if failed:
                print(f"Falhas: {len(failed)}")
                for fail in failed[:3]:  # Mostra apenas primeiras 3 falhas
                    print(f"  - {fail.model}: {fail.error}")
            
            # Mostra algumas respostas
            for result in successful:  # Mostra apenas primeiras 3 respostas
                print(f"\n{result.model}:")
                print(f"  {result.content[:100]}...")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())