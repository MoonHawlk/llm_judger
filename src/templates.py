"""
Templates de prompts para o sistema LLM Judger
"""

from .models import SentencePair


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
