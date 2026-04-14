from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

def web_search(query: str, max_results: int = 5) -> list[dict]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results, region="wt-en"):
            results.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "content": r.get("body"),
                "score": 1.0
            })
    return results

if __name__ == "__main__":
    results = web_search("Groq latest news 2025")
    for r in results:
        print(f"Title: {r['title']}\n   {r['url']}\n")
