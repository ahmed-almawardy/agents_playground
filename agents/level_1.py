#! /usr/bin/python3
# Building a Single-Tool Travel
# Info Agent

import asyncio
import functools
import operator
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

load_dotenv(verbose=True)

llm = ChatOllama(model=os.environ.get("LLM_MODEL", ""), temperature=0.4)
embeddings_model = OllamaEmbeddings(model=os.environ.get("EMBEDDING_MODEL", ""))
data_dir = Path("../data") / "level_1"


vectorstore_client = None
tasks = set()


def run_inew_loop(loop, task, results):
    result = loop.run_until_complete(task)
    results.append(result)
    return result


def run_coro(tasks):
    loop = asyncio.new_event_loop()
    threads = []
    results = []
    for task in tasks:
        thread = threading.Thread(
            target=run_inew_loop, args=[loop, task, results]
        )
        threads.append(thread)
    for thread in threads:
        thread.start()
        thread.join()
    return results


def get_travel_info_vectorstore() -> Chroma:
    global vectorstore_client
    if vectorstore_client is None:
        docs = run_coro([getting_docs(UK_DESTINATIONS)])[0]
        vectorstore_client = build_vectorstore(docs)
    return vectorstore_client


async def getting_docs(from_: list[str]):
    base_url = "https://en.wikivoyage.org/wiki"
    urls = [f"{base_url}/{slug}" for slug in from_]
    headers_template = {"User-Agent": os.environ.get("WIKI_CRAWLER_USER_AGENT")}
    html_loaded = AsyncHtmlLoader(urls, header_template=headers_template)
    return await html_loaded.aload()


def build_vectorstore(docs) -> Chroma:
    """Download WikiVoyage pages and create a Chroma vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=128
    )
    chunks = functools.reduce(
        operator.iadd, [splitter.split_documents([d]) for d in docs], []
    )
    print("Waiting DB..")
    db = get_chroma(chunks)
    db.add_documents(chunks)
    print("DB Done..")
    return db


def get_chroma(chunks):
    vectorstore = Chroma(
        embedding_function=embeddings_model,
        persist_directory=data_dir,
    )
    if not data_dir.exists():
        vectorstore.add_documents(chunks)
    return vectorstore


UK_DESTINATIONS = [
    "Cornwall",
    # "North_Cornwall",
    # "South_Cornwall",
    # "West_Cornwall",
]


vectorstore_client = get_travel_info_vectorstore()
vectorstore_retriever = vectorstore_client.as_retriever()


@tool
def search_travel_info(query: str) -> str:  # B
    """Search embedded WikiVoyage content for
    information about destinations in England."""
    docs = vectorstore_retriever.invoke(query)  # C
    top = docs[:4] if isinstance(docs, list) else docs  # C
    return "\n---\n".join(d.page_content for d in top)  # D


TOOLS = [search_travel_info]
llm = llm.bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class ToolExecutionNode:
    """Execute tools requested by the LLM in the last AIMessage."""

    def __init__(self, tools: Sequence):
        self._tools = {tool.name: tool for tool in tools}

    def __call__(self, agent_state: AgentState):
        messages = agent_state.get("messages", [])
        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool = self._tools.get(tool_name)
            response = tool.invoke(tool_call.get("args"))
            message = ToolMessage(
                tool_call_id=tool_call.get("id"),
                content=response,
                name=tool_name,
            )
            tool_messages.append(message)
        return {"messages": tool_messages}


def llm_node(agent_state: AgentState):
    """LLM node that decides whether to call the search tool."""
    messages = agent_state.get("messages")
    response = llm.invoke(messages)
    return {"messages": [response]}


tool_exec_node = ToolExecutionNode(TOOLS)


agent_graph = StateGraph(AgentState)
agent_graph.add_node("llm_node", llm_node)
agent_graph.add_node("tools", tool_exec_node)
agent_graph.add_conditional_edges("llm_node", tools_condition)
agent_graph.add_edge("llm_node", "tools")
agent_graph.set_entry_point("llm_node")
travel_info_agent = agent_graph.compile()


def chat_loop():  # A
    print("mini Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        state = {"messages": [HumanMessage(content=user_input)]}
        result = travel_info_agent.invoke(state)
        response_msg = result["messages"][-1]
        print(f"Assistant: {response_msg.content}\n")


if __name__ == "__main__":
    chat_loop()
