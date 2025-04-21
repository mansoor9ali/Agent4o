import os
import asyncio
from dotenv import load_dotenv
from crewai.tools import BaseTool
from sympy import sympify  # For safer mathematical evaluation

# Import the ReActAgent class from our Agent4o module
from Agent4o import ReActAgent, configure_azure_openai

load_dotenv()  # Load environment variables from .env

# --- 1. Define CrewAI Tools (Functions the Agent Can Use) ---
class SearchTool(BaseTool):
    """
    A tool for performing web searches.
    """
    name: str = "search_internet"  # Add type annotation
    description: str = "Useful for when you need to find information on the internet. Input should be a specific search query."

    async def _run(self, query: str) -> str:
        """Simulates a web search and returns results."""
        print(f"\n> 🔍 Searching the internet for: '{query}'\n")
        # Replace this with actual search API integration (e.g., Google Search API)
        # Simulate network delay
        await asyncio.sleep(0.5)
        search_results = f"Simulated search results for '{query}': Found some relevant information..."
        return search_results


class CalculatorTool(BaseTool):
    """
    A tool for performing mathematical calculations.
    """
    name: str = "perform_calculation"  # Add type annotation
    description: str = "Useful for solving math problems. Input should be a valid mathematical expression."

    async def _run(self, expression: str) -> str:
        """Performs a calculation using sympy for safety."""
        print(f"\n> 🧮 Calculating: '{expression}'\n")
        try:
            # Simulate computation time
            await asyncio.sleep(0.2)
            result = sympify(expression)  # Safely parse and evaluate the expression
            return f"The result of '{expression}' is: {result}"
        except Exception as e:
            return f"Error: Calculation failed: {e}"

# --- 2. Main Execution ---
async def main():
    try:
        # 2.1 Configure Azure OpenAI and get the model name.
        model_name = await configure_azure_openai()

        # 2.2 Define the tools the agent can use.
        tools = [
            SearchTool(),
            CalculatorTool(),
            # Add more tools here as needed (e.g., for database access, other APIs)
        ]

        # 2.3 Instantiate the ReAct agent using the async factory method
        # You can toggle between CrewAI and direct function calling by setting use_direct_calling
        use_direct_calling = True  # Set to True to use direct function calling, False to use CrewAI
        
        react_agent = await ReActAgent.create(
            name="Agent4o",
            role="Intelligent Assistant",
            goal="Answer user questions accurately and efficiently using available tools.",
            tools=tools,
            llm_model_name=model_name,  # Pass the model name here
            use_direct_calling=use_direct_calling  # Enable direct function calling
        )

        # 2.4 Start the conversation loop.
        while True:
            user_query = input("You: ")
            if user_query.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            response = await react_agent.run_task(user_query)
            print(f"Agent: {response}")
    except Exception as e:
        print(f"Error during initialization: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
