import os
import asyncio
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional, Callable, Union
from openai import AzureOpenAI
from datetime import datetime

# Load environment variables
load_dotenv()

# --- Define utility functions for direct function calling ---
def GetDate():
    """Returns the current date in DD/MM/YYYY format."""
    return datetime.now().strftime("%d/%m/%Y")

def GetTime():
    """Returns the current time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")

def GetCoordinatesOfCity(city_name):
    """
    Fetches the latitude and longitude of a given city name using the Maps.co Geocoding API.

    Args:
    city_name (str): The name of the city.

    Returns:
    tuple: The latitude and longitude of the city.
    """
    base_url = "https://geocode.maps.co/search"
    params = {"q": city_name}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        print(f"Response: {response.json()}")
        data = response.json()
        if data:
            # Assuming the first result is the most relevant
            return data[0]["lat"], data[0]["lon"]
        else:
            return {"error": "No data found for the given city name."}
    else:
        return {"error": "Failed to fetch data, status code: {}".format(response.status_code)}

def GetWeather(city: str) -> dict:
    """
    Retrieves the current weather for a given city.
    
    Parameters:
        city (str): The name of the city.
        
    Returns:
        dict: A dictionary containing weather information.
    """
    import requests
    api_key = "YOUR_API_KEY"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    data = response.json()
    return {
        'location': data['location']['name'],
        'temperature_c': data['current']['temp_c'],
        'condition': data['current']['condition']['text']
    }    

# Function registry for direct function calling
function_registry = {
    "GetDate": GetDate,
    "GetTime": GetTime,
    "GetWeather": GetWeather,
    "GetCoordinatesOfCity": GetCoordinatesOfCity
}

# --- Azure OpenAI Configuration ---
async def configure_azure_openai():
    """Configures environment variables for Azure OpenAI."""
    # Set environment variables for litellm
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Get the model deployment name
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # This should be in format 'azure/deployment-name'
    
    if not all([os.environ.get("AZURE_API_BASE"), 
                os.environ.get("AZURE_API_VERSION"), 
                os.environ.get("AZURE_API_KEY"), 
                model_name]):
        raise ValueError("Azure OpenAI configuration is incomplete. Check your environment variables.")
    
    return model_name

# --- ReActAgent Class Implementation ---
class ReActAgent:
    """
    An agent that uses the ReAct framework to interact with the user and available tools.
    Supports both CrewAI tools and direct function calling with Azure OpenAI.
    """
    def __init__(self, name: str, role: str, goal: str, tools: List[BaseTool] = None):
        """
        Regular constructor (non-async).
        For async initialization, use create() class method instead.
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.tools = tools or []
        self.agent = None
        self.openai_client = None
        self.deployment_name = None
        self.use_direct_calling = False  # Flag to determine which approach to use
        
    @classmethod
    async def create(cls, name: str, role: str, goal: str, tools: List[BaseTool], llm_model_name: str, use_direct_calling: bool = False):
        """
        Asynchronous factory method to create and initialize a ReActAgent.

        Args:
            name (str): The name of the agent.
            role (str): The role of the agent.
            goal (str): The goal of the agent.
            tools (List[BaseTool]): The tools available to the agent.
            llm_model_name (str): The name of the LLM model to use (deployment name).
            use_direct_calling (bool): Whether to use direct function calling instead of CrewAI.
            
        Returns:
            ReActAgent: A fully initialized ReActAgent instance.
        """
        instance = cls(name, role, goal, tools)
        instance.use_direct_calling = use_direct_calling
        
        if use_direct_calling:
            # Setup for direct function calling with Azure OpenAI
            instance.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            # Remove 'azure/' prefix if present
            instance.deployment_name = llm_model_name.replace("azure/", "")
        else:
            # Configure Agent with CrewAI
            instance.agent = Agent(
                role=role,
                goal=goal,
                backstory="You are a helpful assistant that uses the ReAct framework to answer questions.",
                verbose=True,  # Set to True to see the ReAct steps in action
                llm=llm_model_name,  # Pass the model name directly
                tools=tools,
            )
        
        return instance

    async def run_task(self, query: str) -> str:
        """
        Runs a task with the agent.

        Args:
            query (str): The user's query.

        Returns:
            str: The agent's response.
        """
        try:
            if self.use_direct_calling:
                return await self._run_with_function_calling(query)
            else:
                return await self._run_with_crewai(query)
        except Exception as e:
            return f"Error: Task execution failed: {e}"
    
    async def _run_with_crewai(self, query: str) -> str:
        """
        Runs a task using CrewAI.
        """
        # Add expected_output to the Task constructor to comply with newer CrewAI versions
        task = Task(
            description=query,
            agent=self.agent,
            expected_output="A helpful response to the user's query."
        )
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,  # Set verbosity for detailed output
        )
        # Since crew.kickoff() is blocking, run it in a thread to not block the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff)
        return result
    
    async def _run_with_function_calling(self, query: str) -> str:
        """
        Runs a task using direct function calling with Azure OpenAI.
        """
        # Define the tools for function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "GetDate",
                    "description": "Returns the current date.",
                    "parameters": {}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "GetTime",
                    "description": "Returns the current time.",
                    "parameters": {}
                }
            },
            # Can add more functions here as needed
        ]
        
        # Send user prompt to the model
        response = self.openai_client.chat.completions.create(
            model=self.deployment_name,
            temperature=0.5,
            max_tokens=500,
            messages=[{"role": "user", "content": query}],
            tools=tools
        )
        
        # Check if the model wants to call a function
        if response.choices[0].message.tool_calls:
            messages = [{"role": "user", "content": query}]
            
            # Include the assistant's message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function", 
                        "function": {"name": tool_call.function.name, "arguments": "{}"}
                    } 
                    for tool_call in response.choices[0].message.tool_calls
                ]
            })
            
            # Process all tool calls in this response
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                if function_name in function_registry:
                    # Call the function and get the result
                    function_result = function_registry[function_name]()
                    
                    # Add the function result to messages
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id, 
                        "content": function_result
                    })
            
            # Send the function results back for a final response
            final_response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                tools=tools
            )
            
            return final_response.choices[0].message.content
        else:
            # Return direct response if no function calls
            return response.choices[0].message.content
