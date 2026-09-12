#! /usr/bin/python3
# Building a Single-Tool Travel
# Info Agent

import functools
import operator
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict, Literal

import requests
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.agents import create_agent

from agents._helpers import run_coro

load_dotenv(verbose=True)


llm = ChatOllama(model=os.environ.get("LLM_MODEL", ""), temperature=0.4)
embeddings_model = OllamaEmbeddings(model=os.environ.get('EMBEDDING_MODEL', ''))
data_dir = Path("../data") / "level_1"


vectorstore_client = None
tasks = set()



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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = functools.reduce(operator.iadd, [splitter.split_documents([d]) for d in docs], [])
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
def search_travel_info(query: str) -> str:
    """Search embedded WikiVoyage content for
    information about destinations in England."""
    docs = vectorstore_retriever.invoke(query)
    top = docs[:4] if isinstance(docs, list) else docs
    return "\n---\n".join(d.page_content for d in top)


@tool
def get_weather(for_alocation: str, at_datetime: datetime =  None):
    """Get Weather Tool:
        Used to retrieve weather information for a specific location on a specific date. 
        The location can be a town, city, or similar region, followed by the country, separated by a comma.
        if user specificed a date pass it to the tool, if not pass the current date
        Examples:
        Cairo, Egypt
        London, England
        Moscow, Russia
    """
    if not at_datetime: at_datetime = datetime.now()
    url = f"https://api.weatherapi.com/v1/current.json?key={os.environ.get('WEATHERAPI_KEY')}&q={for_alocation}&dt={at_datetime}"
    response = requests.get(url, json=True)
    try:
        response.raise_for_status()
    except:
        return {'error': "can't retrieve the forcast for this location"}
    return response.json()


# class WeatherForecast(TypedDict):
#     town: str
#     weather: Literal["sunny", "foggy", "rainy", "windy"]
#     temperature: int


TOOLS = [search_travel_info, get_weather]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


travel_info_agent = create_agent(
    model=llm,
    state_schema=AgentState,
    tools=TOOLS,
    system_prompt="""You are a helpful assistant
    that can search travel information and get the weather forecast.
    Only use the tools to find the information you need (including town names).""")


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
