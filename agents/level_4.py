#! /usr/bin/python3
# multi-agent architecture

import operator
import os
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.agents import create_agent

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.level_3 import travel_info_agent

load_dotenv(verbose=True)

BASE_DIRE = Path(os.getcwd()).parent

llm = ChatOllama(model=os.environ.get("LLM_MODEL", ""), temperature=0.4)
embeddings_model = OllamaEmbeddings(model=os.environ.get('EMBEDDING_MODEL', ''))
data_dir = Path("../data") / "level_1"
hotel_db = SQLDatabase.from_uri(f'sqlite:///{BASE_DIRE}/data/hotel_data/cornwall_hotels.db')
hotel_db_toolkit = SQLDatabaseToolkit(db=hotel_db, llm=llm)

vectorstore_client = None
tasks = set()


class BnBOffer(TypedDict):
    bnb_id: int
    bnb_name: str
    town: str
    available_rooms: int
    price_per_room: float


class BnBBookingService:
    @staticmethod
    def get_offers_near_town(town: str, num_rooms: int) -> list[BnBOffer]:
        mock_bnb_offers = [
            {
                "bnb_id": 1,
                "bnb_name": "Seaside BnB",
                "town": "Newquay",
                "available_rooms": 3,
                "price_per_room": 80.0,
            },
            {
                "bnb_id": 2,
                "bnb_name": "Surfside Guesthouse",
                "town": "Newquay",
                "available_rooms": 2,
                "price_per_room": 85.0,
            },
            {
                "bnb_id": 3,
                "bnb_name": "Harbour View BnB",
                "town": "Falmouth",
                "available_rooms": 4,
                "price_per_room": 78.0,
            },
            {
                "bnb_id": 4,
                "bnb_name": "Seafarer's Rest",
                "town": "Falmouth",
                "available_rooms": 1,
                "price_per_room": 90.0,
            },
            {
                "bnb_id": 5,
                "bnb_name": "Garden Gate BnB",
                "town": "St Austell",
                "available_rooms": 2,
                "price_per_room": 82.0,
            },
            {
                "bnb_id": 6,
                "bnb_name": "Coastal Cottage BnB",
                "town": "St Austell",
                "available_rooms": 3,
                "price_per_room": 88.0,
            },
            {
                "bnb_id": 7,
                "bnb_name": "Penzance Pier BnB",
                "town": "Penzance",
                "available_rooms": 2,
                "price_per_room": 95.0,
            },
            {
                "bnb_id": 8,
                "bnb_name": "Cornish Charm BnB",
                "town": "Penzance",
                "available_rooms": 3,
                "price_per_room": 87.0,
            },
            {
                "bnb_id": 9,
                "bnb_name": "Camborne Corner BnB",
                "town": "Camborne",
                "available_rooms": 2,
                "price_per_room": 75.0,
            },
            {
                "bnb_id": 10,
                "bnb_name": "Rose Cottage BnB",
                "town": "Camborne",
                "available_rooms": 2,
                "price_per_room": 79.0,
            },
            {
                "bnb_id": 11,
                "bnb_name": "Hayle Haven BnB",
                "town": "Hayle",
                "available_rooms": 3,
                "price_per_room": 83.0,
            },
            {
                "bnb_id": 12,
                "bnb_name": "Dune View BnB",
                "town": "Hayle",
                "available_rooms": 1,
                "price_per_room": 81.0,
            },
            {
                "bnb_id": 13,
                "bnb_name": "Land's End Lookout BnB",
                "town": "Land's End",
                "available_rooms": 2,
                "price_per_room": 100.0,
            },
            {
                "bnb_id": 14,
                "bnb_name": "Atlantic Edge BnB",
                "town": "Land's End",
                "available_rooms": 2,
                "price_per_room": 105.0,
            },
            {
                "bnb_id": 15,
                "bnb_name": "Bude Beach BnB",
                "town": "Bude",
                "available_rooms": 2,
                "price_per_room": 77.0,
            },
            {
                "bnb_id": 16,
                "bnb_name": "Cliffside BnB",
                "town": "Bude",
                "available_rooms": 3,
                "price_per_room": 80.0,
            },
            {
                "bnb_id": 17,
                "bnb_name": "Padstow Harbour BnB",
                "town": "Padstow",
                "available_rooms": 2,
                "price_per_room": 92.0,
            },
            {
                "bnb_id": 18,
                "bnb_name": "Fisherman's Rest BnB",
                "town": "Padstow",
                "available_rooms": 2,
                "price_per_room": 89.0,
            },
            {
                "bnb_id": 19,
                "bnb_name": "St Ives Bay BnB",
                "town": "St Ives",
                "available_rooms": 3,
                "price_per_room": 97.0,
            },
            {
                "bnb_id": 20,
                "bnb_name": "Artists' Retreat BnB",
                "town": "St Ives",
                "available_rooms": 2,
                "price_per_room": 102.0,
            },
            {
                "bnb_id": 21,
                "bnb_name": "Looe Riverside BnB",
                "town": "Looe",
                "available_rooms": 2,
                "price_per_room": 84.0,
            },
            {
                "bnb_id": 22,
                "bnb_name": "Harbour Lights BnB",
                "town": "Looe",
                "available_rooms": 2,
                "price_per_room": 86.0,
            },
            {
                "bnb_id": 23,
                "bnb_name": "Polperro Cove BnB",
                "town": "Polperro",
                "available_rooms": 2,
                "price_per_room": 91.0,
            },
            {
                "bnb_id": 24,
                "bnb_name": "Smuggler's Rest BnB",
                "town": "Polperro",
                "available_rooms": 2,
                "price_per_room": 93.0,
            },
            {
                "bnb_id": 25,
                "bnb_name": "Mevagissey Harbour BnB",
                "town": "Mevagissey",
                "available_rooms": 2,
                "price_per_room": 90.0,
            },
            {
                "bnb_id": 26,
                "bnb_name": "Seafarer's BnB",
                "town": "Mevagissey",
                "available_rooms": 2,
                "price_per_room": 88.0,
            },
            {
                "bnb_id": 27,
                "bnb_name": "Port Isaac View BnB",
                "town": "Port Isaac",
                "available_rooms": 2,
                "price_per_room": 99.0,
            },
            {
                "bnb_id": 28,
                "bnb_name": "Fisherman's Cottage BnB",
                "town": "Port Isaac",
                "available_rooms": 2,
                "price_per_room": 101.0,
            },
            {
                "bnb_id": 29,
                "bnb_name": "Fowey Quay BnB",
                "town": "Fowey",
                "available_rooms": 2,
                "price_per_room": 94.0,
            },
            {
                "bnb_id": 30,
                "bnb_name": "Riverside Rest BnB",
                "town": "Fowey",
                "available_rooms": 2,
                "price_per_room": 96.0,
            },
        ]
        return [
            offer
            for offer in mock_bnb_offers
            if offer["town"].lower() == town.lower()
            and offer["available_rooms"] >= num_rooms
        ]


@tool(
    description="""Check BnB room availability and price for a destination in Cornwall."""
)
def check_bnb_availability(destination: str, num_rooms: int) -> list[dict]:
    offers = BnBBookingService.get_offers_near_town(destination, num_rooms)
    if not offers:
        return [
            {
                "error": f"No available BnBs found in {destination} for {num_rooms} rooms."
            }
        ]
    return offers


BOOKING_TOOLS = hotel_db_toolkit.get_tools() + [check_bnb_availability]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


accommodation_booking_agent = create_agent(
    model=llm,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    system_prompt="""You are a helpful assistant that can check 
    hotel and BnB room availability and price for a
    destination in Cornwall. You can use the tools to 
    get the information you need. If the users does 
    not specify the accommodation type, you should 
    check both hotels and BnBs."""
)


class AgentType(str, Enum):
    travel_info_agent = 'travel_info_agent'
    accommodation_booking_agent = 'accommodation_booking_agent'


class AgentTypeOutput(BaseModel):
    agent: AgentType = Field(..., description="Which agent should handle the question")



ROUTER_SYSTEM_PROMPT = (
    """You are a router. Given the following user message, 
    decide if it is a travel information question 
    (about destinations, attractions, or general travel info) """
    """or an accommodation booking question (about hotels, 
    BnBs, room availability, or prices).\n"""
    """If it is a travel information question, 
    respond with 'travel_info_agent'.\n"""
    """If it is an accommodation booking question, 
    respond with 'accommodation_booking_agent'."""
)


llm_router = llm.with_structured_output(AgentTypeOutput)

def router_agent_node(agent_state: AgentState):
    """Router node: decides which agent should the user query"""
    messages = agent_state['messages']
    last_message = messages[-1] if messages else None
    if isinstance(last_message, HumanMessage):
        user_query = last_message.content
        router_messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]
        response = llm_router.invoke(router_messages)
        return Command(update=agent_state, goto=response.agent.value)
    return Command(update=agent_state, goto=AgentType.travel_info_agent)


router_agent = StateGraph(AgentState)
router_agent.add_node('router_agent', router_agent_node)
router_agent.add_node('travel_info_agent', travel_info_agent)
router_agent.add_node('accommodation_booking_agent', accommodation_booking_agent)
router_agent.add_edge('travel_info_agent', END)
router_agent.add_edge('accommodation_booking_agent', END)
router_agent.set_entry_point('router_agent')
router_agent = router_agent.compile()




def chat_loop():  # A
    print("mini Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        state = {"messages": [HumanMessage(content=user_input)]}
        result = router_agent.invoke(state)
        response_msg = result["messages"][-1]
        print(f"Assistant: {response_msg.content}\n")




if __name__ == "__main__":
    chat_loop()
