import asyncio
import os
import uuid
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.level_3 import travel_info_agent
from agents.level_4 import accommodation_booking_agent

llm = ChatOllama(model=os.environ.get("LLM_MODEL"))


class AgentState(MessagesState):
    pass


GUARDRAIL_SYSTEM_PROMPT = """You are a strict classifier. Given the user's 
    last message, respond with whether it is
    travel-related . Travel-related queries 
    include destinations, attractions, lodging (hotels/BnBs), 
    room availability, prices, or weather in Cornwall/England.
    """


class GuardrailOutput(BaseModel):
    is_travel: bool = Field(
        ...,
        description=(
            """True if the user question is about travel information: 
            destinations, attractions, 
            lodging (hotels/BnBs), prices, availability, 
            or weather in Cornwall/England."""
        ),
    )
    reason: str = Field(..., description="Brief justification for the decision")


llm_guardrail = ChatOllama(
    model=os.environ.get("LLM_MODEL")
).with_structured_output(GuardrailOutput)


class AgentType(str, Enum):
    travel_agent_node = "travel_agent_node"
    accommodation_agent_node = "accommodation_agent_node"


class AgentTypeOutput(BaseModel):
    agent: AgentType = Field(
        ..., description="Which agent should handle the query?"
    )


llm_router = ChatOllama(
    model=os.environ.get("LLM_MODEL")
).with_structured_output(AgentTypeOutput)


ROUTER_SYSTEM_PROMPT = """You are a router. Given the following user message, 
    decide if it is a travel information question 
    (about destinations, attractions, or general travel info) 
    or an accommodation booking question (about hotels, BnBs, 
    room availability, or prices).\n
    If it is a travel information question, respond with 
    'travel_agent_node'.\n
    If it is an accommodation booking question, 
    respond with 'accommodation_agent_node'."""


def router_agent_node(agent_state: AgentState):
    last_message = agent_state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        classifier_messages = [
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
            last_message,
        ]
        response = llm_guardrail.invoke(classifier_messages)
        if not response.is_travel:
            refusal_text = """Sorry, I can only help with travel-related 
                questions (destinations, attractions, lodging, 
                prices, availability, or weather in Cornwall/England). 
                Please rephrase your request to be travel-related."""
            return Command(  # D
                update={"messages": [AIMessage(content=refusal_text)]},
                goto="guardrail_refusal_node",
            )
        router_messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            last_message,
        ]
        response = llm_router.invoke(router_messages)
        return Command(goto=response.agent.value)
    return Command(goto=AgentType.travel_agent_node.value)


def guardrail_refusal_node(agent_state: AgentState):
    return {}


def get_agent():
    graph = StateGraph(AgentState)
    graph.add_node("router_agent_node", router_agent_node)
    graph.add_node("travel_agent_node", travel_info_agent)
    graph.add_node("accommodation_agent_node", accommodation_booking_agent)
    graph.add_node("guardrail_refusal_node", guardrail_refusal_node)
    graph.set_entry_point("router_agent_node")
    graph.add_edge("travel_agent_node", END)
    graph.add_edge("accommodation_agent_node", END)
    graph.add_edge("guardrail_refusal_node", END)

    return graph.compile(checkpointer=InMemorySaver())


async def main():
    config = {"configurable": {"thread_id": uuid.uuid8()}}
    agent = get_agent()
    print('Welcome to MultiModal System "quit" to exit')
    print(f"Thread ID: {config['configurable']['thread_id']}")
    while True:
        human_message = HumanMessage(content=input("You -> ").strip())
        response = await agent.ainvoke({"messages": [human_message]}, config)
        print(f"AI -> {response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
