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

# LangChain community imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agents import AgentExecutor, Tool  # Updated import
from langchain_community.chains.summarize import load_summarize_chain
from langchain_community.memory import ConversationBufferWindowMemory

# LangChain core imports
from langchain_core.prompts import PromptTemplate, SystemMessage
from langchain_core.text_splitter import RecursiveCharacterTextSplitter

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
        'X-API-KEY': 'ab179d0f00ae0bafe47f77e09e62b9f53b3f281d',
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
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    
    map_prompt_template = PromptTemplate(
        template="Write a summary of the following text:\n\"{text}\"\nSUMMARY:",
        input_variables=["text"]
    )
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )
    
    return summary_chain.run(input_documents=docs)

def research(query):
    """Perform comprehensive research on a topic."""
    
    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; you do not make things up, you will try as hard as possible to gather facts & data to back up the research. Please make sure you complete the objective above with the following rules: 
1/ You will always search for internal knowledge base first to see if there are any relevant information 
2/ If the internal knowledge doesn't have good result, then you can go search online 
3/ While searching online: 
   a/ You will try to collect as many useful details as possible 
   b/ If there are URLs of relevant links & articles, you will scrape it to gather more information 
   c/ After scraping & searching, you should think "is there any new things I should search & scrape based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations 
4/ You should not make things up; you should only write facts & data that you have gathered 
5/ In the final output, you should include all reference data & links to back up your research"""
    )

    tools = [
        Tool(
            name="Knowledge_retrieval",
            func=knowledge_retrieval,
            description="Use this to get our internal knowledge base data for curated information, always use this first before searching online"
        ),
        Tool(
            name="Google_search",
            func=search,
            description="Always use this to answer questions about current events, data, or terms that you don't really understand. You should ask targeted questions"
        ),
        Tool(
            name="Scrape_website",
            func=scrape_website,
            description="Use this to load content from a website URL"
        ),
    ]

    # Initialize AgentExecutor instead of initialize_agent
    agent_executor = AgentExecutor(
        agent=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=tools,
        verbose=False,
        agent_kwargs={"system_message": system_message}
    )

    return agent_executor.run(query)

def create_agent(id, user_name, ai_name, instructions):
    """Create a new agent with memory and tools."""
    
    system_message = SystemMessage(content=instructions)
    
    memory = ConversationBufferWindowMemory(
        memory_key="memory",
        return_messages=True,
        ai_prefix=ai_name,
        user_prefix=user_name
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
            description="Use this to load content from a website URL"
        ),
    ]

    # Initialize AgentExecutor instead of initialize_agent for new agents too
    agent_executor = AgentExecutor(
        agent=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=tools,
        verbose=True,
        agent_kwargs={"system_message": system_message},
        memory=memory
    )

    agents[id] = agent_executor  # Store the created agent in global agents dictionary
    
    return agent_executor

def generate_response(instructions, user_input):
    """Generate response using appropriate agent."""
    
    id = user_input["id"]
    message = user_input["message"]
    
    if id not in agents:
        agent_executor = create_agent(id, user_input["user_name"], user_input["ai_name"], instructions)
    else:
        agent_executor = agents[id]
    
    return agent_executor.run(message)

def generate_gpt4_response(prompt):
    """Generate response using GPT-4o-mini model."""
    
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{"role": "system", "name": "admin_user", "content": prompt}]
    )
    
    return response.choices[0].message.content

async def poly_image_gen(session, prompt):
    """Generate image using Pollinations AI."""
    
    seed = random.randint(1, 100000)
    
    image_url = f"https://image.pollinations.ai/prompt/{prompt}?seed={seed}"
    
    async with session.get(image_url) as response:
         image_data = await response.read()
         
         return io.BytesIO(image_data)

async def dall_e_gen(model, prompt, size, num_images):
     """Generate images using DALL-E."""
     
     response = openai.Image.create(
         model=model,
         prompt=prompt,
         n=num_images,
         size=size,
     )
     
     async with aiohttp.ClientSession() as session:
         tasks = []
         for image in response["data"]:
             async with session.get(image["url"]) as response:
                 content = await response.content.read()
                 tasks.append(io.BytesIO(content))
                 
         return tasks

async def generate_image_prodia(prompt, model, sampler, seed, neg=None):
     """Generate image using Prodia API."""
     
     print("\033[1;32m(Prodia) Creating image for:\033[0m", prompt)
     start_time = time.time()

     async def create_job(prompt, model, sampler, seed, neg):
         negative = neg if neg else "(nsfw:1.5),verybadimagenegative_v1.3,..."

         params = {
             'new': 'true',
             'prompt': quote(prompt),
             'model': model,
             'negative_prompt': negative,
             'steps': '100',
             'cfg': '9.5',
             'seed': str(seed),
             'sampler': sampler,
             'upscale': 'True',
             'aspect_ratio': 'square'
         }

         async with aiohttp.ClientSession() as session:
             async with session.get('https://api.prodia.com/generate', params=params) as response:
                 data = await response.json()
                 return data['job']

     job_id = await create_job(prompt, model, sampler, seed, neg)

     headers = {
         'authority': 'api.prodia.com',
         'accept': '*/*',
     }

     async with aiohttp.ClientSession() as session:
         while True:
             async with session.get(f'https://api.prodia.com/job/{job_id}', headers=headers) as response:
                 json_response = await response.json()
                 if json_response['status'] == 'succeeded':
                     async with session.get(f'https://images.prodia.xyz/{job_id}.png?download=1', headers=headers) as img_response:
                         content = await img_response.content.read()
                         img_file_obj = io.BytesIO(content)
                         duration = time.time() - start_time
                         print(f"\033[1;34m(Prodia) Finished image creation\n\033[0mJob id: {job_id} Prompt: {prompt} in {duration} seconds.")
                         return img_file_obj

