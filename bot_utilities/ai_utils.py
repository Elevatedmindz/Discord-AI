# Standard library imports
import aiohttp
import io
from datetime import datetime
import time
import random
from urllib.parse import quote
import os
import json
import requests

# Third-party imports
from dotenv import find_dotenv, load_dotenv
import openai
from bs4 import BeautifulSoup
from pydantic import Field

# LangChain core imports
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chains import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local imports
from bot_utilities.config_loader import load_current_language, config

# Initialize environment and configurations
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
current_language = load_current_language()
internet_access = config['INTERNET_ACCESS']

# Global variables
agents = {}

def sdxl(prompt):
    """Generate image using SDXL model."""
    response = openai.Image.create(
        model="sdxl", prompt=prompt, n=1, size="1024x1024"
    )
    return response['data'][0]["url"]

def knowledge_retrieval(query):
    """Retrieve knowledge from the API."""
    data = {
        "params": {"query": query},
        "project": "feda14180b9d-4ba2-9b3c-6c721dfe8f63"
    }
    response = requests.post(
        "https://api-1e3042.stack.tryrelevance.com/latest/studios/6eba417b-f592-49fc-968d-6b63702995e3/trigger_limited",
        data=json.dumps(data)
    )
    if response.status_code == 200:
        return response.json()["output"]["answer"]
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return None

def scrape_website(url: str):
    """Scrape website content and summarize if too large."""
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data = {"url": url}
    response = requests.post(
        "https://chrome.browserless.io/content?token=0a049e5b-3387-4c51-ab6c-57647d519571",
        headers=headers, data=json.dumps(data)
    )
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        return summary(text) if len(text) > 10000 else text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return None

def search(query):
    """Perform Google search using Serper API."""
    headers = {
        'X-API-KEY': os.getenv('SERPER_API_KEY'),  # Use environment variable instead
        'Content-Type': 'application/json'
    }
    response = requests.post(
        "https://google.serper.dev/search",
        headers=headers,
        data=json.dumps({"q": query})
    )
    return response.json()

def summary(content):
    """Generate summary of large text content."""
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    
    map_prompt_template = PromptTemplate(
        template="Write a summary of the following text:\n\"{text}\"\nSUMMARY:",
        input_variables=["text"]
    )
    
    return load_summarize_chain(
        llm=llm, chain_type='map_reduce', verbose=True
    ).run(input_documents=docs)

def research(query):
    """Perform comprehensive research on a topic."""
    tools = [
        Tool(
            name="Knowledge_retrieval",
            func=knowledge_retrieval,
            description="Use this to get curated information from the internal knowledge base."
        ),
        Tool(
            name="Google_search",
            func=search,
            description="Use this to perform a Google search."
        ),
        Tool(
            name="Scrape_website",
            func=scrape_website,
            description="Use this to scrape content from a URL."
        ),
    ]
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent.run(query)

def create_agent(id, user_name, ai_name, instructions):
    """Create a new agent with memory and tools."""
    memory = ConversationBufferMemory(
        memory_key="memory",
        return_messages=True
    )

    tools = [
        Tool(
            name="research",
            func=research,
            description="Perform comprehensive research."
        ),
        Tool(
            name="Scrape_website",
            func=scrape_website,
            description="Scrape content from a URL."
        ),
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    agents[id] = agent
    return agent

def generate_response(instructions, user_input):
    """Generate response using the appropriate agent."""
    id = user_input["id"]
    message = user_input["message"]
    agent = agents.get(id) or create_agent(
        id, user_input["user_name"], user_input["ai_name"], instructions
    )
    return agent.run(message)
