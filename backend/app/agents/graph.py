from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes import execute_agent_node, fallback_agent_node, research_agent_node
from app.agents.state import SOFT_LOOP_LIMIT, AgentState

workflow = StateGraph(AgentState)

workflow.add_node("research_agent", research_agent_node)
workflow.add_node("execute_agent", execute_agent_node)
workflow.add_node("fallback_agent", fallback_agent_node)

workflow.set_entry_point("research_agent")


def _is_loop_limit_reached(state: AgentState) -> bool:
  return state.get("loop_count", 0) > SOFT_LOOP_LIMIT


def research_router(state: AgentState) -> str:
  if _is_loop_limit_reached(state):
    return "fallback"

  query = state.get("user_query", "").lower()
  if "thực thi" in query or "chuyển tiền" in query:
    return "execute"

  return "end"


def post_execute_router(state: AgentState) -> str:
  if _is_loop_limit_reached(state):
    return "fallback"

  if state.get("needs_review"):
    return "research"

  return "end"


workflow.add_conditional_edges(
  "research_agent",
  research_router,
  {
    "execute": "execute_agent",
    "end": END,
    "fallback": "fallback_agent",
  },
)

workflow.add_conditional_edges(
  "execute_agent",
  post_execute_router,
  {
    "research": "research_agent",
    "end": END,
    "fallback": "fallback_agent",
  },
)

workflow.add_edge("fallback_agent", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
