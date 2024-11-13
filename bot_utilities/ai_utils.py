import aiohttp
import io
from datetime import datetime
import time
import random
from urllib.parse import quote
from bot_utilities.config_loader import load_current_language, config
import openai
import os
from dotenv import find_dotenv, load_dotenv
import json

from langchain_community.agent_toolkits import create_python_agent
from langchain_community.agents import AgentType, Tool, initialize_agent
from langchain_community.chains import LLMMathChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from bs4 import BeautifulSoup
from pydantic import Field
import requests

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
current_language = load_current_language()
internet_access = config['INTERNET_ACCESS']

def create_agent(id, user_name, ai_name, instructions):
    system_message = SystemMessage(
        content=instructions
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, ai_prefix=ai_name, user_prefix=user_name)

    # Update to use GPT-4o-mini model
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini",
        openai_api_key=os.getenv("CHIMERA_GPT_KEY"),
        openai_api_base="https://api.naga.ac/v1"
    )
    
    tools = [                     
        Tool(
            name="research",
            func=research,
            description="Always use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
        ),           
        Tool(
            name="Scrape_website",
            func=scrape_website,
            description="Use this to load content from a website url"
        ),   
    ]    

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory
    )

    agents[id] = agent
    return agent

def generate_response(instructions, user_input):   
    id = user_input["id"]    
    message = user_input["message"]

    if id not in agents:
        user_name = user_input["user_name"]
        ai_name = user_input["ai_name"]
        agent = create_agent(id, user_name, ai_name, instructions)
    else:
        agent = agents[id]
    
    print(message)
    response = agent.run(message)
    return response

def generate_response_old(instructions, search, history):
    if search is not None:
        search_results = search
    elif search is None:
        search_results = "Search feature is disabled"
    messages = [
        {"role": "system", "name": "instructions", "content": instructions},
        *history,
        {"role": "system", "name": "search_results", "content": search_results},
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        api_base="https://api.naga.ac/v1",
        api_key=os.getenv("CHIMERA_GPT_KEY")
    )
    message = response.choices[0].message.content
    return message

# Rest of your existing functions remain the same...
