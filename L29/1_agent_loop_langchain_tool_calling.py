from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = 'qwen3:1.7b'

# -- Tools (LangChain @tool decorator) --

@tool
def get_product_price(product : str) -> float:
    """
    Look up the price of a product in the catalog.
    """
    print(f"    >> Executing get_product_price=(product='{product}')")
    prices = {"laptop":1299.99, "headphones": 149.95, "keyboard" : 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price : float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available Tiers: bronze, silver, gold"""
    print(f">> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    
    discount_percentages = {"bronze":float(5),"silver":float(12), "gold": float(23)}
    discount = discount_percentages.get(discount_tier)
    return round(price * (1-discount / 100)) # type: ignore

@traceable(name='LangChain Agent Loop')
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name : t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" *60)
    
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f'\n--- Iteration {iteration}  ---')

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f'\nFinal Answer: {ai_message.content}')
            return ai_message.content

        # process only the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args',{})
        tool_call_id = tool_call.get('id')

        print(f'  [Tool Selected] {tool_name}')

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        observation = tool_to_use.invoke(tool_args)

        print(f". [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print('ERROR: Max iterations reached without a final answer')

if __name__ == '__main__':
    print('Hello Langchain Agent (.bind_tools)')
    print()
    result = run_agent('What is the price of a laptop after applying a gold discount?')
