from dotenv import load_dotenv

load_dotenv()

# from langchain import hub
# from langchain.agents import AgentExecutor
# from langchain.agents.react.agent import create_react_agent
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
