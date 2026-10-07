from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel): 
    '''Schema for a source used by the agent'''

    url:str = Field(description='The url of the source')

class AgentResponse(BaseModel):
    '''Schema for agent response with answer and sources'''

    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources to generate the answer")

llm = ChatOpenAI(model='gpt-5')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print('Hello from langchain-course')
    result = agent.invoke(
        {
            'messages' : HumanMessage(
                content='search for 3 job postings for an ai engineer using langchain in Cincinnati, OH on linkedin and list their details'
            )
        }
    )
    print(result)

if __name__ == '__main__':
    main()