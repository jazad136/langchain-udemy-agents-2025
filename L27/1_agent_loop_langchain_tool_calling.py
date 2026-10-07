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
    return prices.get(product, 4)

def apply_discount(price : float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available Tiers: bronze, silver, gold"""
    print(f">> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze":5,"silver":12, "gold": 23}
    discount = discount_percentages.get(discount_tier)
    return discount
@traceable(name='LangChain Agent Loop')
def run_agent(question: str):
    pass

if __name__ == '__main__':
    print('Hello Langchain Agent (.bind_tools)')
    print()
    result = run_agent('What is the price of a laptop after applying a gold discount?')
