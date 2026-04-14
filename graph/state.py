from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # The original user query
    query: str

    # Full message history across agents
    messages: Annotated[List[BaseMessage], operator.add]

    # Raw search results from Search Agent
    search_results: List[dict]

    # Analyzed/synthesized content from Analysis Agent
    analysis: str

    # Final written report from Writer Agent
    report: str

    # Feedback from Critic Agent
    critique: str

    # Whether the output passed the critic's check
    is_approved: bool

    # Which agent should act next
    next_agent: str

    # How many revision cycles have happened (to prevent infinite loops)
    revision_count: int