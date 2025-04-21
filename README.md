# Agent4o: Azure OpenAI Agent Framework

Agent4o is a Python-based agent framework that leverages Azure OpenAI services to create intelligent agents capable of using tools and function calling. This framework provides both high-level abstractions for AI agents as well as direct function calling capabilities with robust error handling and fallback mechanisms.

## Features

- **Dual Operating Modes**: 
  - CrewAI-based agent capabilities for complex multi-agent tasks
  - Direct Function Calling with Azure OpenAI for efficient single-agent interactions

- **Built-in Utility Functions**:
  - Date and time retrieval (`GetDate`, `GetTime`)
  - Geographic coordinates lookup with geocoding API (`GetCoordinatesOfCity`)
  - Weather data retrieval with real-time information (`GetWeatherData`)
  - Automatic mock data generation when APIs are unavailable
  - Easy to extend with custom functions

- **Advanced Features**:
  - Intelligent function chaining for multi-step queries
  - Smart fallback responses when models encounter issues
  - Comparison capabilities for weather between multiple locations
  - Robust error handling throughout the pipeline

- **Modular Architecture**:
  - Clean separation of agent logic and application code
  - Easily extendable for new capabilities
  - Comprehensive function registry system

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # On Windows
   ```
3. Install dependencies:
   ```
   pip install python-dotenv crewai openai sympy requests
   ```
4. Copy `.env.sample` to `.env` and add your Azure OpenAI credentials

## Configuration

Create a `.env` file with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
AZURE_OPENAI_KEY="your_azure_openai_api_key"
AZURE_OPENAI_DEPLOYMENT_NAME="azure/deployment_name"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

## Usage

### Basic Agent Usage

```python
import asyncio
from Agent4o import ReActAgent, configure_azure_openai

async def main():
    # Configure Azure OpenAI
    model_name = await configure_azure_openai()
    
    # Create the agent
    agent = await ReActAgent.create(
        name="MyAgent",
        role="Assistant",
        goal="Help users with their questions",
        tools=[],  # CrewAI tools if needed
        llm_model_name=model_name,
        use_direct_calling=True  # Use direct function calling
    )
    
    # Run simple queries
    date_response = await agent.run_task("What is the current date and time?")
    print(f"Date/Time Response: {date_response}\n")
    
    # Get weather information
    weather_response = await agent.run_task("What's the weather like in New York City?")
    print(f"Weather Response: {weather_response}\n")
    
    # Run complex multi-function queries
    complex_response = await agent.run_task("Compare the weather in London and Paris right now")
    print(f"Complex Response: {complex_response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Adding Custom Functions

You can add custom functions to the function registry in `Agent4o.py`:

```python
def MyCustomFunction(param1, param2):
    """
    Description of what your function does.
    Provide clear documentation as this will be used by the model.
    """
    # Implementation with robust error handling
    try:
        # Your function logic here
        result = process_data(param1, param2)
        return result
    except Exception as e:
        print(f"Error in MyCustomFunction: {str(e)}")
        return {"error": str(e), "fallback_data": "Some default value"}

# Add to the function registry
function_registry = {
    "GetDate": GetDate,
    "GetTime": GetTime,
    "GetCoordinatesOfCity": GetCoordinatesOfCity,
    "GetWeatherData": GetWeatherData,
    "MyCustomFunction": MyCustomFunction
}
```

### Adding Function Definitions for the Model

For best results, add function definitions to the `_run_with_function_calling` method in the ReActAgent class:

```python
# Add this to the tools list
{
    "type": "function",
    "function": {
        "name": "MyCustomFunction",
        "description": "Description of what your function does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of parameter 1"
                },
                "param2": {
                    "type": "integer",
                    "description": "Description of parameter 2"
                }
            },
            "required": ["param1", "param2"]
        }
    }
}
```

## Testing

The framework includes a comprehensive test suite that validates all functionality across multiple categories:

- Basic Functionality (date, time)
- Location Services (coordinates lookup)
- Weather Information (temperature, wind speed)
- Multi-function Queries (combining multiple functions)
- Comparison Features (comparing data between locations)

Run the test suite to verify all capabilities:

```
python test_Agent4o.py
```

The test script provides detailed output showing the system's performance across all test categories, including execution times and success rates.

## Reliability Features

Agent4o includes several features to ensure reliability in production environments:

- **API Fallbacks**: Automatic fallback to mock data when external APIs are unavailable
- **Error Handling**: Comprehensive error handling throughout the framework
- **Response Validation**: Smart fallback response generation when the model struggles
- **Realistic Mock Data**: Geographic and weather mock data based on real-world patterns
- **Performance Metrics**: Built-in timing for monitoring execution performance

## Example Applications

- **Virtual Assistants**: Create assistants that can provide real-time information
- **Data Analysis**: Build agents that can retrieve and compare data from multiple sources
- **Planning Systems**: Develop agents that can consider weather and location in planning
- **Customer Support**: Implement agents that can handle complex multi-step inquiries

## License

[MIT License](LICENSE)
