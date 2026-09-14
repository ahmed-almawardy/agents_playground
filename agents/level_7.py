import asyncio
import os
import sys
import uuid
from enum import Enum
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.level_3 import travel_info_agent
from agents.level_4 import (
    ROUTER_SYSTEM_PROMPT,
    accommodation_booking_agent,
)


class AgentState(MessagesState):
    pass


class AgentTypes(str, Enum):
    travel_info_agent = "travel_info_agent"
    accommodation_booking_agent = "accommodation_booking_agent"


class AgentTypeOutput(BaseModel):
    agent: AgentTypes = Field(
        ..., description="Which agent should handle the query?"
    )


router_llm = ChatOllama(
    model=os.environ.get("LLM_MODEL")
).with_structured_output(AgentTypeOutput)


def router_agent(agent_state: AgentState):
    last_message = agent_state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=last_message.content),
        ]

        response = router_llm.invoke(messages)
        return Command(update=agent_state, goto=response.agent.value)
    return Command(update=agent_state, goto=AgentTypes.travel_info_agent)


def draw_graph():
    graph = StateGraph(state_schema=AgentState)
    graph.add_node("router_agent", router_agent)
    graph.add_node("travel_info_agent", travel_info_agent)
    graph.add_node("accommodation_booking_agent", accommodation_booking_agent)
    graph.add_edge("travel_info_agent", END)
    graph.add_edge("accommodation_booking_agent", END)
    graph.set_entry_point("router_agent")
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


async def chat():
    agent = draw_graph()
    thread_session_id = uuid.uuid8()
    print(f"Session ID: {thread_session_id}")
    config = {"configurable": {"thread_id": thread_session_id}}
    print(
        "Welcome to AI smart multimodal Graph (type 'quit' or 'exit') to quit"
    )
    while True:
        human_message = HumanMessage(content=input("You # ").strip())
        if human_message.content in {"exit", "quit"}:
            sys.exit(0)
        agent_state = {"messages": [human_message]}
        response = await agent.ainvoke(agent_state, config)
        print(f"AI # {response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(chat())
