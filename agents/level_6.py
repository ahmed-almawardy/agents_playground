import asyncio
import operator
import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from agents.level_3 import search_travel_info

load_dotenv(verbose=True)


async def get_accuweather_tools() -> list[BaseTool]:
    """Consuming MCP servers"""
    mcp_client = MultiServerMCPClient(
        {
            "weatherapi": {
                "url": "http://127.0.0.1:8020/weatherapi-mcp-server",
                "transport": "streamable_http",
            }
        }
    )
    return await mcp_client.get_tools()


messages = []


class AgentState(TypedDict):  # A
    messages: Annotated[Sequence[BaseMessage], operator.add]


async def chat_loop(agent):
    print("Welcome to smart AI loop (to quit write 'quit' or 'exit')")
    messages_dict = {"messages": messages}
    while True:
        human_message = input("You > ").strip()
        if human_message in {"quit", "exit"}:
            print("Bye...")
            break
        messages.append(HumanMessage(content=human_message))
        response_messages_dict = await agent.ainvoke(messages_dict)
        print(f"AI > {response_messages_dict['messages'][-1].content}")
        messages.append(response_messages_dict["messages"][-1])


async def main():
    """:::MainRun:::"""
    tools: list[BaseTool] = await get_accuweather_tools()
    tools.append(search_travel_info)
    agent = create_agent(
        model=ChatOllama(model=os.environ.get("LLM_MODEL")),
        state_schema=AgentState,
        name="agent_assistant",
        tools=tools,
        system_prompt="""You are a helpful assistant that can 
    search travel information and get the weather forecast. 
    Only use the tools to find the information you need 
    (including town names).""",
    )
    await chat_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
