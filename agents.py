from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_scraper,web_search
import os
from dotenv import load_dotenv

load_dotenv()

model= ChatGroq(model="openai/gpt-oss-120b",temperature=0)

#agent1
def build_search_agent():
    return create_agent(
        model=model,
        tools=[web_search]
    )

#agent 2
def build_reader_agent():
    return create_agent(
        model=model,
        tools=[web_scraper]
    )

#chain 1
writer_prompt=ChatPromptTemplate.from_messages([
    ("system","you are an expert research writer. write clear,structured and insightful reports"),
    ("human","""write a detailed research report on the topic given below.
    
    topic:{topic}

    research gathered:{research}

    structure the report as:
    -introduction
    -key finding(minimun 3 well explained points)
    -conclusion
    -source(list all urls found in the research)

    be detailed, factual and professional""")

])

str_parser=StrOutputParser()

writer_chain=writer_prompt| model| str_parser

#chain 2 (critic chain to give review)
critic_prompt=ChatPromptTemplate.from_messages([
    ("system","you are a sharp and constructive research critic. be honest and specific."),
    ("human","""review the research report below and evaluate it strictly
    
    report:{report}
    respond in this exact format:

    score:X/10
    strengths:
    -...
    -...
    areas to improve:
    -...
    -...
    one line verdict:...
    """)
])

critic_chain=critic_prompt|model|str_parser

