from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

SOFT_LOOP_LIMIT = 4


class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage], add_messages]
  user_query: str
  current_agent: str
  last_agent: str
  needs_review: bool
  loop_count: Annotated[int, lambda old, new: old + new]
