import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.graph import app_graph

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
  message: str = Field(min_length=1)
  thread_id: str | None = None


class ApproveRequest(BaseModel):
  thread_id: str
  approved: bool


def _build_initial_state(message: str) -> dict[str, Any]:
  return {
    "messages": [HumanMessage(content=message)],
    "user_query": message,
    "current_agent": "",
    "last_agent": "",
    "needs_review": False,
    "loop_count": 0,
  }


def _serialize_messages(messages: list[Any]) -> list[dict[str, str]]:
  serialized: list[dict[str, str]] = []
  for message in messages:
    role = "assistant"
    if isinstance(message, HumanMessage):
      role = "user"
    elif isinstance(message, AIMessage):
      role = "assistant"
    serialized.append({"role": role, "content": message.content})
  return serialized


def _format_graph_result(result: dict[str, Any]) -> dict[str, Any]:
  interrupts = result.get("__interrupt__")
  pending_approval = None
  if interrupts:
    pending_approval = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]

  return {
    "thread_id": result.get("thread_id"),
    "messages": _serialize_messages(result.get("messages", [])),
    "current_agent": result.get("current_agent"),
    "loop_count": result.get("loop_count", 0),
    "needs_review": result.get("needs_review", False),
    "pending_approval": pending_approval,
    "is_paused": pending_approval is not None,
  }


@router.post("")
def chat(request: ChatRequest) -> dict[str, Any]:
  thread_id = request.thread_id or str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}

  result = app_graph.invoke(_build_initial_state(request.message), config)
  response = _format_graph_result(result)
  response["thread_id"] = thread_id
  return response


@router.post("/approve")
def approve(request: ApproveRequest) -> dict[str, Any]:
  config = {"configurable": {"thread_id": request.thread_id}}
  snapshot = app_graph.get_state(config)

  if not snapshot.next:
    raise HTTPException(status_code=400, detail="Không có luồng nào đang chờ phê duyệt.")

  result = app_graph.invoke(Command(resume=request.approved), config)
  response = _format_graph_result(result)
  response["thread_id"] = request.thread_id
  return response
