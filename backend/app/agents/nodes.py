from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.agents.state import AgentState


def research_agent_node(state: AgentState) -> dict:
  user_query = state.get("user_query", "")
  print(f"--- ENTER RESEARCH AGENT (Loop Count: {state.get('loop_count', 0)}) ---")

  ai_reply = (
    f"Hệ thống đã tìm thấy thông tin về: '{user_query}'. "
    "Bạn có muốn thực thi lệnh chuyển tiền/xuất báo cáo không?"
  )

  return {
    "messages": [AIMessage(content=ai_reply)],
    "current_agent": "research_agent",
    "last_agent": "research_agent",
    "needs_review": False,
    "loop_count": 1,
  }


def execute_agent_node(state: AgentState) -> dict:
  print("--- ENTER EXECUTE AGENT (CRITICAL ZONE) ---")

  approved = interrupt(
    {
      "type": "approval_required",
      "user_query": state.get("user_query", ""),
      "message": "Cần phê duyệt trước khi thực thi giao dịch nhạy cảm.",
    }
  )

  if not approved:
    return {
      "messages": [AIMessage(content="Giao dịch đã bị từ chối sau khi xem xét.")],
      "current_agent": "execute_agent",
      "last_agent": "execute_agent",
      "needs_review": False,
      "loop_count": 1,
    }

  user_query = state.get("user_query", "").lower()
  needs_review = "xem lại" in user_query

  ai_reply = "Thực thi hành động nhạy cảm thành công! Giao dịch của bạn đã được ghi nhận."
  if needs_review:
    ai_reply += " Hệ thống sẽ quay lại bước nghiên cứu để xem lại kết quả."

  return {
    "messages": [AIMessage(content=ai_reply)],
    "current_agent": "execute_agent",
    "last_agent": "execute_agent",
    "needs_review": needs_review,
    "loop_count": 1,
  }


def fallback_agent_node(state: AgentState) -> dict:
  print("--- ENTER FALLBACK AGENT (LOOP LIMIT REACHED) ---")

  return {
    "messages": [
      AIMessage(
        content=(
          f"Cảnh báo: Hệ thống dừng do vượt giới hạn {state.get('loop_count', 0)} vòng lặp "
          f"(tối đa cho phép). Vui lòng thử lại với yêu cầu rõ ràng hơn hoặc liên hệ hỗ trợ."
        )
      )
    ],
    "current_agent": "fallback_agent",
    "last_agent": "fallback_agent",
    "needs_review": False,
    "loop_count": 0,
  }
