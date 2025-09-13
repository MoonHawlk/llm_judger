"""
Modelos de dados para o sistema LLM Judger
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuração de modelo para julgamento"""
    name: str
    instances: int
    temperature: float = 0.1  # Baixo para consistência em julgamentos
    max_tokens: int = 512


@dataclass
class SentencePair:
    """Par de sentenças para julgamento"""
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    context: Optional[str] = None
    reference_id: Optional[str] = None


@dataclass
class JudgmentResponse:
    """Resposta de julgamento do modelo"""
    model: str
    sentence_pair: SentencePair
    is_correct: bool
    confidence_score: float
    reasoning: str
    timestamp: datetime
    success: bool
    error: Optional[str] = None


@dataclass
class CSVRow:
    """Linha de dados CSV"""
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    context: Optional[str] = None
    row_index: int = 0
