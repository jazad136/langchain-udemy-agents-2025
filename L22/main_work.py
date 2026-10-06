from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

llm = ChatOpenAI(model='gpt-5')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

@tool
def search(query : str):
    """
    Tool that searches over internet
    Args: 
        query: The query to search for
    Returns: 
        The search result
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


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