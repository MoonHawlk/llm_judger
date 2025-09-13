"""
Processadores de dados para o sistema LLM Judger
"""

import logging
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path
from .models import SentencePair, JudgmentResponse

logger = logging.getLogger(__name__)


class CSVProcessor:
    """Processador de datasets CSV para julgamento"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        self.source_col: Optional[str] = None
        self.target_col: Optional[str] = None
        self.source_lang: str = "auto"
        self.target_lang: str = "auto"
        
    def load_csv(self) -> bool:
        """Carrega o arquivo CSV e valida formato"""
        try:
            if not self.csv_path.exists():
                logger.error(f"Arquivo não encontrado: {self.csv_path}")
                return False
            
            # Tenta diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    logger.info(f"CSV carregado com encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                logger.error("Não foi possível carregar o CSV com nenhum encoding")
                return False
            
            logger.info(f"CSV carregado: {len(self.df)} linhas, {len(self.df.columns)} colunas")
            logger.info(f"Colunas disponíveis: {list(self.df.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV: {e}")
            return False
    
    def validate_columns(self, source_col: str, target_col: str) -> bool:
        """Valida se as colunas especificadas existem"""
        if self.df is None:
            return False
        
        if source_col not in self.df.columns:
            logger.error(f"Coluna fonte '{source_col}' não encontrada")
            return False
        
        if target_col not in self.df.columns:
            logger.error(f"Coluna alvo '{target_col}' não encontrada")
            return False
        
        self.source_col = source_col
        self.target_col = target_col
        
        logger.info(f"Colunas configuradas: {source_col} -> {target_col}")
        return True
    
    def set_languages(self, source_lang: str, target_lang: str):
        """Define os idiomas das colunas"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        logger.info(f"Idiomas configurados: {source_lang} -> {target_lang}")
    
    def get_sentence_pairs(self) -> List[SentencePair]:
        """
        Extrai pares de sentenças do DataFrame carregado
        
        Returns:
            List[SentencePair]: Lista de pares de sentenças
        
        Raises:
            ValueError: Se as colunas não foram configuradas ou CSV não foi carregado
        """
        if self.df is None:
            raise ValueError("CSV não foi carregado. Execute load_csv() primeiro.")
        
        if not self.source_col or not self.target_col:
            raise ValueError("Colunas não configuradas. Execute validate_columns() primeiro.")
        
        sentence_pairs = []
        
        # Método mais seguro: usar enumerate para ter controle total sobre o índice
        for row_idx, (pandas_idx, row) in enumerate(self.df.iterrows()):
            try:
                # Usar row_idx que é sempre um inteiro sequencial (0, 1, 2, ...)
                row_number = row_idx + 1
                
                source_text = str(row[self.source_col]).strip()
                target_text = str(row[self.target_col]).strip()
                
                # Pula linhas vazias ou com valores nan
                if (not source_text or not target_text or 
                    source_text.lower() == 'nan' or target_text.lower() == 'nan' or
                    source_text == '' or target_text == ''):
                    logger.warning(f"Linha {row_number}: texto vazio ou nan - pulando")
                    continue
                
                sentence_pair = SentencePair(
                    source_text=source_text,
                    target_text=target_text,
                    source_language=self.source_lang,
                    target_language=self.target_lang,
                    reference_id=f"csv_row_{row_number}"
                )
                
                sentence_pairs.append(sentence_pair)
                
            except Exception as e:
                # Usar row_idx + 1 para evitar problemas com pandas_idx
                logger.error(f"Erro ao processar linha {row_idx + 1}: {e}")
                continue
        
        logger.info(f"Extraídos {len(sentence_pairs)} pares válidos de sentenças")
        return sentence_pairs
    
    def save_results(self, results: List[JudgmentResponse], output_path: Optional[str] = None) -> bool:
        """Salva os resultados no CSV original ou em novo arquivo"""
        try:
            if self.df is None:
                logger.error("DataFrame não carregado")
                return False
            
            # Cria cópia do DataFrame original
            result_df = self.df.copy()
            
            # Inicializa colunas de resultado se não existirem
            if 'resultado' not in result_df.columns:
                result_df['resultado'] = ''
            if 'explicacao' not in result_df.columns:
                result_df['explicacao'] = ''
            if 'confianca' not in result_df.columns:
                result_df['confianca'] = 0.0
            if 'modelo' not in result_df.columns:
                result_df['modelo'] = ''
            if 'timestamp' not in result_df.columns:
                result_df['timestamp'] = ''
            
            # Create a mapping from position to DataFrame index
            position_to_df_index = {}
            valid_pairs_count = 0
            
            for position, (df_idx, row) in enumerate(self.df.iterrows(), start=1):
                source_text = str(row[self.source_col]).strip()
                target_text = str(row[self.target_col]).strip()
                
                # Skip empty rows (same logic as get_sentence_pairs)
                if not source_text or not target_text or source_text == 'nan' or target_text == 'nan':
                    continue
                
                valid_pairs_count += 1
                position_to_df_index[valid_pairs_count] = df_idx
            
            # Mapeia resultados para as linhas correspondentes
            for result in results:
                # Extrai número da linha do reference_id
                if result.sentence_pair.reference_id and result.sentence_pair.reference_id.startswith('csv_row_'):
                    try:
                        position = int(result.sentence_pair.reference_id.split('_')[-1])
                        
                        # Get the actual DataFrame index for this position
                        if position in position_to_df_index:
                            df_idx = position_to_df_index[position]
                            
                            result_df.loc[df_idx, 'resultado'] = 'Correto' if result.is_correct else 'Incorreto'
                            result_df.loc[df_idx, 'explicacao'] = result.reasoning
                            result_df.loc[df_idx, 'confianca'] = result.confidence_score
                            result_df.loc[df_idx, 'modelo'] = result.model
                            result_df.loc[df_idx, 'timestamp'] = result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            logger.warning(f"Position {position} não encontrada no mapeamento")
                            
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Não foi possível mapear resultado: {result.sentence_pair.reference_id} - {e}")
            
            # Define caminho de saída
            if output_path is None:
                # Adiciona sufixo ao arquivo original
                output_path = self.csv_path.parent / f"{self.csv_path.stem}_resultados{self.csv_path.suffix}"
            
            # Salva o arquivo
            result_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Resultados salvos em: {output_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo do dataset"""
        if self.df is None:
            return {}
        
        return {
            "total_rows": len(self.df),
            "columns": list(self.df.columns),
            "source_column": self.source_col,
            "target_column": self.target_col,
            "source_language": self.source_lang,
            "target_language": self.target_lang
        }
