"""
Utilitários para o sistema LLM Judger
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from .models import JudgmentResponse


def setup_logging(debug_mode: bool = False) -> None:
    """Configura o sistema de logging"""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def print_results_summary(results: List[JudgmentResponse]) -> None:
    """Imprime resumo dos resultados de julgamento"""
    if not results:
        print("Nenhum resultado para exibir.")
        return
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n=== Resumo dos Resultados ===")
    print(f"Total de julgamentos: {len(results)}")
    print(f"Sucessos: {len(successful)}")
    print(f"Falhas: {len(failed)}")
    
    if successful:
        correct_count = sum(1 for r in successful if r.is_correct)
        avg_confidence = sum(r.confidence_score for r in successful) / len(successful)
        
        print(f"\nEstatísticas dos Sucessos:")
        print(f"- Correto: {correct_count}/{len(successful)} ({correct_count/len(successful)*100:.1f}%)")
        print(f"- Confiança média: {avg_confidence:.2f}")
        
        # Agrupa por modelo
        model_stats = {}
        for result in successful:
            if result.model not in model_stats:
                model_stats[result.model] = {"total": 0, "correct": 0, "confidence": []}
            model_stats[result.model]["total"] += 1
            if result.is_correct:
                model_stats[result.model]["correct"] += 1
            model_stats[result.model]["confidence"].append(result.confidence_score)
        
        print(f"\nEstatísticas por Modelo:")
        for model, stats in model_stats.items():
            accuracy = stats["correct"] / stats["total"] * 100
            avg_conf = sum(stats["confidence"]) / len(stats["confidence"])
            print(f"- {model}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%) - Confiança: {avg_conf:.2f}")


def group_results_by_sentence_pair(results: List[JudgmentResponse]) -> Dict[str, List[JudgmentResponse]]:
    """Agrupa resultados por par de sentenças"""
    pair_results = {}
    
    for result in results:
        if result.success:
            # Cria uma chave única baseada no conteúdo das sentenças
            pair_key = f"{result.sentence_pair.source_text[:50]}... | {result.sentence_pair.target_text[:50]}..."
            if pair_key not in pair_results:
                pair_results[pair_key] = []
            pair_results[pair_key].append(result)
    
    return pair_results


def print_detailed_results(results: List[JudgmentResponse]) -> None:
    """Imprime resultados detalhados agrupados por par de sentenças"""
    pair_results = group_results_by_sentence_pair(results)
    
    if not pair_results:
        print("Nenhum resultado válido para exibir.")
        return
    
    print(f"\n=== Resultados Detalhados ===")
    
    for pair_key, judgments in pair_results.items():
        correct_count = sum(1 for j in judgments if j.is_correct)
        avg_confidence = sum(j.confidence_score for j in judgments) / len(judgments)
        
        print(f"\n--- {pair_key} ---")
        print(f"Julgamentos: {correct_count}/{len(judgments)} corretos")
        print(f"Confiança média: {avg_confidence:.2f}")
        
        # Mostra reasoning de alta confiança
        high_confidence = [j for j in judgments if j.confidence_score > 0.8]
        if high_confidence:
            print(f"Reasoning (alta confiança): {high_confidence[0].reasoning[:200]}...")
        
        # Mostra estatísticas por modelo para este par
        model_stats = {}
        for judgment in judgments:
            if judgment.model not in model_stats:
                model_stats[judgment.model] = {"correct": 0, "total": 0}
            model_stats[judgment.model]["total"] += 1
            if judgment.is_correct:
                model_stats[judgment.model]["correct"] += 1
        
        print("Por modelo:")
        for model, stats in model_stats.items():
            accuracy = stats["correct"] / stats["total"] * 100
            print(f"  - {model}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")


def format_duration(start_time: datetime, end_time: datetime) -> str:
    """Formata duração entre dois timestamps"""
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    
    if total_seconds < 60:
        return f"{total_seconds:.2f}s"
    elif total_seconds < 3600:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"
