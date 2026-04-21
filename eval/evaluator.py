from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field
from utils.groq_llm import chat_groq

load_dotenv()
llm = chat_groq()


class ReportEvaluation(BaseModel):
    relevance: int = Field(ge=1, le=10, description='Does the report answer the query?')
    completeness: int = Field(ge=1, le=10, description='Important points covered?')
    clarity: int = Field(ge=1, le=10, description='Easy to understand?')
    structure: int = Field(ge=1, le=10, description='Well organized?')
    overall: int = Field(ge=1, le=10, description='Overall quality.')
    feedback: str = Field(default='', description='One or two sentences of concise feedback.')


def _fallback_scores_from_text(query: str, report: str, sources: str) -> dict:
    response = llm.invoke([
        SystemMessage(
            content='You are an objective evaluator. Score the report on each dimension from 1-10. Return ONLY valid JSON, nothing else. Format: {"relevance": 8, "completeness": 7, "clarity": 9, "structure": 8, "overall": 8, "feedback": "brief feedback here"}'
        ),
        HumanMessage(
            content=f'Query: {query}\n\nSources used:\n{sources}\n\nReport:\n{report}\n\nScore this report on: relevance (does it answer the query?), completeness (covers all important points?), clarity (easy to understand?), structure (well organized?), overall.'
        ),
    ])
    try:
        text = response.content.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            'relevance': 0,
            'completeness': 0,
            'clarity': 0,
            'structure': 0,
            'overall': 0,
            'feedback': 'Evaluation failed',
        }


def evaluate_report(query: str, report: str, search_results: list) -> dict:
    print('\nEvaluator: scoring report...')

    sources = '\n'.join([r['title'] for r in search_results[:5]])
    human = HumanMessage(
        content=f'Query: {query}\n\nSources used:\n{sources}\n\nReport:\n{report}\n\nScore each dimension 1-10 using the schema.'
    )
    system = SystemMessage(
        content='You are an objective evaluator. Assign integer scores 1-10 for relevance, completeness, clarity, structure, and overall. Add brief feedback.'
    )

    structured_llm = llm.with_structured_output(ReportEvaluation)
    try:
        out = structured_llm.invoke([system, human])
        if isinstance(out, ReportEvaluation):
            scores = out.model_dump()
        else:
            scores = ReportEvaluation.model_validate(out).model_dump()
    except Exception as e:
        print(f'   Structured evaluation failed ({e}); using JSON fallback.')
        scores = _fallback_scores_from_text(query, report, sources)

    print('\n' + '=' * 50)
    print('EVALUATION SCORES')
    print('=' * 50)
    print(f"  Relevance:     {scores.get('relevance', '?')}/10")
    print(f"  Completeness:  {scores.get('completeness', '?')}/10")
    print(f"  Clarity:       {scores.get('clarity', '?')}/10")
    print(f"  Structure:     {scores.get('structure', '?')}/10")
    print(f"  Overall:       {scores.get('overall', '?')}/10")
    print(f"  Feedback:      {scores.get('feedback', '?')}")
    print('=' * 50)

    return scores
