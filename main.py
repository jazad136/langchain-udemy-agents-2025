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


# from langchain.agents import create_agent
# from langchain.agents import AgentExecutor
# from langchain_openai import ChatOpenAI
# from langchain_tavily import TavilySearch
# from langchain import hub

# from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-5")
react_prompt = hub.pull('hwchase17/react')


agent = create_react_agent(
    llm = llm,
    tools=tools,
    prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor
def main():
    result = chain.invoke(
        input = {
            "input" : """
            search for 3 job postings for an 
            ai engineer using langchain in the
            bay area on linkedin and 
            list their details"""
        }
    )
    print(result)

if __name__ == "__main__":
    main()
