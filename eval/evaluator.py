from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def evaluate_report(query: str, report: str, search_results: list) -> dict:
    print('\nEvaluator: scoring report...')

    sources = '\n'.join([r['title'] for r in search_results[:5]])

    response = llm.invoke([
        SystemMessage(content='You are an objective evaluator. Score the report on each dimension from 1-10. Return ONLY valid JSON, nothing else. Format: {"relevance": 8, "completeness": 7, "clarity": 9, "structure": 8, "overall": 8, "feedback": "brief feedback here"}'),
        HumanMessage(content=f'Query: {query}\n\nSources used:\n{sources}\n\nReport:\n{report}\n\nScore this report on: relevance (does it answer the query?), completeness (covers all important points?), clarity (easy to understand?), structure (well organized?), overall.')
    ])

    try:
        text = response.content.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        scores = json.loads(text[start:end])
    except Exception as e:
        print(f'   Failed to parse scores: {e}')
        scores = {'relevance': 0, 'completeness': 0, 'clarity': 0, 'structure': 0, 'overall': 0, 'feedback': 'Evaluation failed'}

    print('\n' + '='*50)
    print('EVALUATION SCORES')
    print('='*50)
    print(f"  Relevance:     {scores.get('relevance', '?')}/10")
    print(f"  Completeness:  {scores.get('completeness', '?')}/10")
    print(f"  Clarity:       {scores.get('clarity', '?')}/10")
    print(f"  Structure:     {scores.get('structure', '?')}/10")
    print(f"  Overall:       {scores.get('overall', '?')}/10")
    print(f"  Feedback:      {scores.get('feedback', '?')}")
    print('='*50)

    return scores
