from dotenv import load_dotenv

load_dotenv()

# from langchain import hub
# from langchain.agents import AgentExecutor
# from langchain.agents.react.agent import create_react_agent
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

# from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
# from langchain.agents.structured_output import ResponseFormat

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# from langchain.agents import create_agent
# from langchain.agents import AgentExecutor
# from langchain_openai import ChatOpenAI
# from langchain_tavily import TavilySearch
# from langchain import hub

# from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
    ,input_variables=["input", "agent=scratchpad", "tool_names"]
).partial(format_instructions="")

# agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x : x["output"])


chain = agent_executor | extract_output | structured_llm


def main():
    result = chain.invoke(
        input={
            "input": """
            search for 3 job postings for an 
            ai engineer using langchain in the
            bay area on linkedin and 
            list their details"""
        }
    )
    print(result)


if __name__ == "__main__":
    main()
