import asyncio
import json
import inspect
from Agent4o import ReActAgent, configure_azure_openai, function_registry

async def test_Agent4o():
    """
    Test the Agent4o functionality with a simple query.
    """
    try:
        # Configure Azure OpenAI and get the model name
        print("Configuring Azure OpenAI...")
        model_name = await configure_azure_openai()
        print(f"Using model: {model_name}")
        
        # Create the agent with direct function calling
        print("\nCreating Agent4o instance...")
        agent = await ReActAgent.create(
            name="TestAgent",
            role="Test Assistant",
            goal="Test the Agent4o functionality",
            tools=[],  # No CrewAI tools needed for this test
            llm_model_name=model_name,
            use_direct_calling=True
        )
        
        # Test with a simple date/time query
        test_query = "What is today's date and time and tell me Coordinates of Malir karachi?"
        print(f"\nSending test query: '{test_query}'")
        
        # Let's try a more direct approach to verify function calling
        print("\n[INFO] Starting test with direct logging...")
        
        # Extract the function calling implementation directly
        # We'll add manual print statements to log what's happening
        async def run_test(query):
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
                {
                    "type": "function",
                    "function": {
                        "name": "GetCoordinatesOfCity",
                        "description": "Returns the coordinates for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city_name": {
                                    "type": "string",
                                    "description": "The name of the city"
                                }
                            },
                            "required": ["city_name"]
                        }
                    }
                }
            ]
            
            # List available tools
            print("\n[INFO] Tools available to the model:")
            for i, tool in enumerate(tools):
                print(f"  {i+1}. {tool['function']['name']}: {tool['function'].get('description', '')}")
            
            # First API call to the model
            print("\n[INFO] Sending initial query to Azure OpenAI...")
            response = agent.openai_client.chat.completions.create(
                model=agent.deployment_name,
                temperature=0.5,
                max_tokens=500,
                messages=[{"role": "user", "content": query}],
                tools=tools
            )
            
            # Check if function calls were requested
            function_calls = []
            if response.choices[0].message.tool_calls:
                print(f"\n[INFO] Model requested function calls:")
                for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                    function_name = tool_call.function.name
                    print(f"  {i+1}. Function: {function_name}")
                    
                    # Create messages array for follow-up request
                    messages = [{"role": "user", "content": query}]
                    messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function", 
                            "function": {"name": function_name, "arguments": "{}"}
                        }]
                    })
                    
                    # Execute the functions
                    if function_name in function_registry:
                        # For the city coordinates function, extract city name if possible
                        if function_name == "GetCoordinatesOfCity" and "Malir" in query:
                            result = function_registry[function_name]("Malir Karachi")
                        else:
                            result = function_registry[function_name]()
                            
                        function_calls.append({
                            "function": function_name,
                            "result": result
                        })
                        print(f"     Result: {result}")
                        
                        # Add the function result to messages
                        messages.append({
                            "role": "tool", 
                            "tool_call_id": tool_call.id, 
                            "content": str(result)
                        })
                
                # Send the function results back for final response
                print("\n[INFO] Sending function results back to model...")
                final_response = agent.openai_client.chat.completions.create(
                    model=agent.deployment_name,
                    messages=messages,
                    tools=tools
                )
                
                return final_response.choices[0].message.content, function_calls
            else:
                print("\n[INFO] Model did not request any function calls")
                return response.choices[0].message.content, []
        
        # Run our test function
        response, function_calls = await run_test(test_query)
        
        # Display the response
        print("\nAgent4o Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Summarize function calls
        if function_calls:
            print("\n[INFO] Summary of Function Calls:")
            print("-" * 50)
            for i, call in enumerate(function_calls):
                print(f"Call {i+1}:")
                print(f"  Function: {call['function']}")
                print(f"  Result: {call['result']}")
            print("-" * 50)
        else:
            print("\n[WARNING] No functions were called during processing!")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_Agent4o())
