import os

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph_supervisor.supervisor import create_supervisor

from agents.level_3 import travel_info_agent
from agents.level_4 import accommodation_booking_agent


travel_info_agent.name = 'travel_info_agent'
accommodation_booking_agent.name = 'accommodation_booking_agent'

travel_assistant = create_supervisor(
    agents=[travel_info_agent, accommodation_booking_agent],
    model=ChatOllama(model=os.environ.get('LLM_MODEL')),
    supervisor_name='travel_assistant',
    prompt=(
        """You are a supervisor that manages two agents: 
        a travel information agent and an accommodation booking agent. """
        """You can answer user questions that might require 
        calling both agents when needed. """
        """Decide which agent(s) to use for each user request 
        and coordinate their responses."""
    )
    ).compile()


def chat_loop(): #A
    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip() #B
        if user_input.lower() in {"exit", "quit"}: #C
            break
        state = {"messages": [HumanMessage(content=user_input)]} #D
        result = travel_assistant.invoke(state) #E
        response_msg = result["messages"][-1] #F
        print(f"Assistant: {response_msg.content}\n") #G


if __name__ == "__main__":
    chat_loop()