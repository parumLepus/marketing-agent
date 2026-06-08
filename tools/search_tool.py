from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

_search = DuckDuckGoSearchRun()

@tool
def search_marketing_trends(query: str) -> str:
    """
    Search the web for marketing trends, competitor activity, industry benchmarks,
    or content ideas. Use this to enrich recommendations with current market context.
    Example queries: 'SaaS email marketing benchmarks 2024',
    'Instagram Reels engagement rates B2B'.
    """
    try:
        return _search.run(query)
    except Exception as e:
        return f"Search failed: {e}"