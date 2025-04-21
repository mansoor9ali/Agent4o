import os
import asyncio
import json
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
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
    Returns the coordinates (latitude, longitude) for a given city name.
    Uses the Maps.co Geocoding API or returns mock data if API is unavailable.
    """
    try:
        # First try the actual API
        # URL encode the city name
        encoded_city = city_name.replace(" ", "+")
        
        # Make the API request
        base_url = f"https://geocode.maps.co/search?q={encoded_city}&format=json"
        response = requests.get(base_url)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                # Get the coordinates from the first result
                lat = data[0].get('lat')
                lon = data[0].get('lon')
                print(f"Successfully fetched coordinates for {city_name}: {lat}, {lon}")
                return (lat, lon)
            else:
                print(f"No coordinates found for {city_name}, using mock data")
                # Fall back to mock data
        else:
            print(f"Error fetching coordinates: {response.status_code}, using mock data instead")
            # Fall back to mock data
    except Exception as e:
        print(f"Exception in GetCoordinatesOfCity: {str(e)}, using mock data instead")
        # Fall back to mock data
    
    # Mock data for common cities
    mock_coordinates = {
        "new york": ("40.7128", "-74.0060"),
        "london": ("51.5074", "-0.1278"),
        "paris": ("48.8566", "2.3522"),
        "tokyo": ("35.6762", "139.6503"),
        "sydney": ("-33.8688", "151.2093"),
        "berlin": ("52.5200", "13.4050"),
        "moscow": ("55.7558", "37.6173"),
        "beijing": ("39.9042", "116.4074"),
        "cairo": ("30.0444", "31.2357"),
        "rio": ("-22.9068", "-43.1729"),
        "karachi": ("24.8607", "67.0011"),
        "malir karachi": ("24.8949", "67.2053"),
    }
    
    # Normalize city name for lookup
    normalized_city = city_name.lower()
    
    # Try to find an exact match
    if normalized_city in mock_coordinates:
        return mock_coordinates[normalized_city]
    
    # Try to find a partial match
    for city, coords in mock_coordinates.items():
        if city in normalized_city or normalized_city in city:
            return coords
    
    # Default coordinates if no match found (0°N, 0°E - null island)
    return ("0", "0")

def GetWeatherData(coordinates):
    """
    Returns weather data for given coordinates.
    Uses the Open-Meteo API or mock data if API is unavailable.
    """
    try:
        lat, lon = coordinates
        
        # Check if we have null coordinates (0,0) - use mock data in that case
        if lat == "0" and lon == "0":
            return _get_mock_weather()
        
        # Make the API request
        base_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        response = requests.get(base_url)
        
        if response.status_code == 200:
            data = response.json()
            # Extract current weather data
            if 'current' in data:
                current = data['current']
                unit = 'C'  # Default unit if not available
                
                # Safely get the temperature unit without non-ASCII characters
                if 'current_units' in data and 'temperature_2m' in data['current_units']:
                    unit_str = data['current_units']['temperature_2m']
                    # Replace any potential unicode degree symbol with plain ASCII
                    unit = unit_str.replace('\u00b0', '').strip()
                    if not unit:
                        unit = 'C'  # Default to Celsius if empty
                
                result = {
                    'temperature': current.get('temperature_2m'),
                    'wind_speed': current.get('wind_speed_10m'),
                    'unit': unit
                }
                print(f"Successfully fetched weather data: {result}")
                return result
            else:
                print("No current weather data available, using mock data")
                return _get_mock_weather(lat, lon)
        else:
            print(f"Error fetching weather data: {response.status_code}, using mock data")
            return _get_mock_weather(lat, lon)
    except Exception as e:
        print(f"Exception in GetWeatherData: {str(e)}, using mock data")
        return _get_mock_weather(lat, lon)

def _get_mock_weather(lat=None, lon=None):
    """
    Internal helper to generate realistic mock weather data based on coordinates.
    """
    import random
    from datetime import datetime
    
    # Generate realistic weather data based on latitude (rough approximation)
    # Northern latitudes in winter are colder, etc.
    current_month = datetime.now().month
    is_winter_north = current_month in [12, 1, 2, 3]
    is_summer_north = current_month in [6, 7, 8, 9]
    
    if lat and lon:
        # Convert lat to float if possible
        try:
            lat_float = float(lat)
            # Using latitude to roughly determine temperature range
            # Equatorial (close to 0 degrees latitude)
            if -23.5 <= lat_float <= 23.5:
                temp_base = 28  # Hot tropical climate
                temp_variation = 5
            # Mid-latitudes
            elif 23.5 < abs(lat_float) <= 45:
                if (lat_float > 0 and is_winter_north) or (lat_float < 0 and not is_winter_north):
                    temp_base = 10  # Cooler in winter
                else:
                    temp_base = 22  # Warmer in summer
                temp_variation = 8
            # High latitudes
            else:
                if (lat_float > 0 and is_winter_north) or (lat_float < 0 and not is_winter_north):
                    temp_base = 0  # Cold in winter
                else:
                    temp_base = 15  # Mild in summer
                temp_variation = 10
        except ValueError:
            # Default if lat can't be converted to float
            temp_base = 20
            temp_variation = 10
    else:
        # Default values if no coordinates provided
        temp_base = 20
        temp_variation = 10
        
    # Generate random but plausible values
    temperature = round(temp_base + (random.random() * 2 - 1) * temp_variation, 1)
    wind_speed = round(random.uniform(0, 10), 1)
    
    return {
        'temperature': temperature,
        'wind_speed': wind_speed,
        'unit': 'C'
    }

# Function registry for direct function calling
function_registry = {
    "GetDate": GetDate,
    "GetTime": GetTime,
    "GetCoordinatesOfCity": GetCoordinatesOfCity,
    "GetWeatherData": GetWeatherData
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

    def _generate_fallback_response(self, query, function_results, location_data):
        """
        Generate a fallback response based on function results when the model fails to provide one.
        """
        print("\n[DEBUG] Generating fallback response")
        print(f"  Location data: {location_data}")
        print(f"  Function results: {function_results}")
        
        response_parts = []
        
        # Check if we have date/time information
        date_result = None
        time_result = None
        weather_result = None
        coordinates_result = None
        
        # Extract results from the function calls based on known function names
        for call_id, result in function_results.items():
            # First clean up the result if it's a string
            if isinstance(result, str):
                # Remove any quotes if they encompass the whole string
                result = result.strip()
                if (result.startswith("'") and result.endswith("'")) or \
                   (result.startswith('"') and result.endswith('"')):
                    result = result[1:-1]
                
                # Try to parse dictionaries if they are strings
                if result.startswith('{') and result.endswith('}'): 
                    try:
                        import ast
                        result = ast.literal_eval(result)
                    except:
                        pass
            
            # Check function name from call_id
            if 'GetDate' in call_id:
                date_result = result
            elif 'GetTime' in call_id:
                time_result = result
            elif 'GetWeatherData' in call_id:
                weather_result = result  
            elif 'GetCoordinatesOfCity' in call_id:
                coordinates_result = result
            # Also check the content if function name is unclear
            elif result is not None:
                result_str = str(result).lower()
                if '/date' in result_str and not date_result:
                    date_result = result
                elif ':' in result_str and len(result_str) < 10 and not time_result:
                    time_result = result
                elif 'temperature' in result_str and not weather_result:
                    weather_result = result
                elif isinstance(result, tuple) and len(result) == 2 and not coordinates_result:
                    # Looks like coordinates (lat, lon)
                    coordinates_result = result
        
        # Build a response based on available data
        if 'compare' in query.lower() and ('weather' in query.lower() or 'temperature' in query.lower()):
            # This is a comparison query - look for multiple cities
            print("  Detected comparison query for multiple cities")
            
            # Extract potential city names from the query
            import re
            cities_mentioned = []
            
            # Try to extract cities from query using common patterns
            cities_pattern = r'(?:between|in|of|for)\s+([A-Za-z\s]+)\s+(?:and|&|vs\.?)\s+([A-Za-z\s]+)'
            city_matches = re.search(cities_pattern, query)
            
            if city_matches:
                city1 = city_matches.group(1).strip()
                city2 = city_matches.group(2).strip()
                cities_mentioned = [city1, city2]
                print(f"  Extracted cities from comparison: {cities_mentioned}")
            
            # If we can't find cities with regex, check the coordinates we have
            if not cities_mentioned and len(function_results) >= 2:
                # Try to determine cities from multiple coordinate calls
                potential_cities = []
                for call_id, result in function_results.items():
                    if 'GetCoordinatesOfCity' in call_id:
                        # Try to extract city from arguments
                        for msg in function_results.values():
                            if isinstance(msg, str) and 'coordinates' in msg.lower():
                                # This might be city name
                                potential_cities.append("a location")
                
                if len(potential_cities) >= 2:
                    cities_mentioned = potential_cities[:2]
                    print(f"  Inferred cities from function calls: {cities_mentioned}")
            
            # If we still don't have cities, use some default ones from coordinates
            if not cities_mentioned:
                # Default to comparing some common cities
                cities_mentioned = ["the first location", "the second location"]
                print(f"  Using default city names: {cities_mentioned}")
            
            # Prepare comparison response
            response_parts.append(f"Here's the weather comparison you requested:\n\n")
            
            # Get weather data for each city
            city_weather_data = []
            
            # Process each city
            for city in cities_mentioned:
                city_data = {"name": city}
                
                # Check if we have coordinates for this city
                city_coords = None
                
                # Look for these coordinates in our function results
                for call_id, result in function_results.items():
                    if 'GetCoordinatesOfCity' in call_id and city.lower() in call_id.lower():
                        try:
                            if isinstance(result, str):
                                import ast
                                coords_tuple = ast.literal_eval(result)
                                city_coords = coords_tuple
                            elif isinstance(result, tuple):
                                city_coords = result
                        except:
                            pass
                
                # If we don't have coordinates but have a city name, try to get them
                if not city_coords and city != "a location" and city != "the first location" and city != "the second location":
                    try:
                        city_coords = GetCoordinatesOfCity(city)
                        print(f"  Got coordinates for {city}: {city_coords}")
                    except:
                        pass
                
                # If we have coordinates, get weather data
                if city_coords:
                    try:
                        weather_data = GetWeatherData(city_coords)
                        city_data["weather"] = weather_data
                        city_data["coordinates"] = city_coords
                    except Exception as e:
                        print(f"  Error getting weather for {city}: {e}")
                
                city_weather_data.append(city_data)
            
            # Now build the comparison response
            for i, city_data in enumerate(city_weather_data):
                city_name = city_data.get("name")
                weather = city_data.get("weather")
                
                response_parts.append(f"**{city_name}**:\n")
                
                if weather:
                    # Format the weather data
                    try:
                        if isinstance(weather, dict):
                            temp = weather.get('temperature')
                            wind = weather.get('wind_speed')
                            unit = weather.get('unit', 'C')
                            response_parts.append(f"- Temperature: {temp}°{unit}\n")
                            response_parts.append(f"- Wind Speed: {wind} m/s\n")
                        else:
                            response_parts.append(f"- Weather: {weather}\n")
                    except:
                        response_parts.append(f"- Weather data: {weather}\n")
                else:
                    response_parts.append(f"- Weather data not available\n")
                
                response_parts.append("\n")
            
            # Add comparison conclusion
            if len(city_weather_data) >= 2 and all("weather" in city for city in city_weather_data):
                try:
                    temp1 = city_weather_data[0]["weather"].get("temperature")
                    temp2 = city_weather_data[1]["weather"].get("temperature")
                    
                    if temp1 and temp2 and isinstance(temp1, (int, float)) and isinstance(temp2, (int, float)):
                        temp_diff = abs(float(temp1) - float(temp2))
                        warmer_city = city_weather_data[0]["name"] if float(temp1) > float(temp2) else city_weather_data[1]["name"]
                        cooler_city = city_weather_data[1]["name"] if float(temp1) > float(temp2) else city_weather_data[0]["name"]
                        
                        response_parts.append(f"**Comparison**: {warmer_city} is {temp_diff:.1f}°C warmer than {cooler_city} right now.\n")
                except Exception as e:
                    print(f"  Error generating comparison conclusion: {e}")
            
        elif 'weather' in query.lower() or 'temperature' in query.lower():
            # This is a regular weather query for a single location
            city_name = location_data.get('city', 'the location')
            response_parts.append(f"Here's the information you requested about {city_name}:\n")
            
            if date_result:
                response_parts.append(f"Date: {date_result}\n")
            
            if time_result:
                response_parts.append(f"Time: {time_result}\n")
            
            # Check if we have weather data in the location_data as a backup 
            if not weather_result and 'coordinates' in location_data:
                print("  Fetching weather data from coordinates in location_data")
                try:
                    weather_data = GetWeatherData(location_data['coordinates'])
                    weather_result = weather_data
                    print(f"  Generated weather data: {weather_data}")
                except Exception as e:
                    print(f"  Error getting weather from coordinates: {e}")
            
            if weather_result:
                print(f"  Weather result type: {type(weather_result)}, value: {weather_result}")
                try:
                    if isinstance(weather_result, dict):
                        temp = weather_result.get('temperature')
                        wind = weather_result.get('wind_speed')
                        unit = weather_result.get('unit', 'C')
                        response_parts.append(f"Weather:\n- Temperature: {temp}\u00b0{unit}\n- Wind Speed: {wind} m/s\n")
                    elif isinstance(weather_result, str) and '{' in weather_result:
                        # Try to parse as dictionary
                        import ast
                        try:
                            weather_dict = ast.literal_eval(weather_result)
                            temp = weather_dict.get('temperature')
                            wind = weather_dict.get('wind_speed')
                            unit = weather_dict.get('unit', 'C')
                            response_parts.append(f"Weather:\n- Temperature: {temp}\u00b0{unit}\n- Wind Speed: {wind} m/s\n")
                        except:
                            response_parts.append(f"Weather: {weather_result}\n")
                    else:
                        response_parts.append(f"Weather: {weather_result}\n")
                except Exception as e:
                    print(f"  Error formatting weather result: {e}")
                    response_parts.append(f"Weather: {weather_result}\n")
            
            if coordinates_result and not weather_result:
                response_parts.append(f"I found coordinates {coordinates_result} but couldn't retrieve weather data.\n")
                
        elif 'date' in query.lower() or 'time' in query.lower():
            # This is a date/time query
            response_parts.append("Here's the current date and time:\n")
            
            if date_result:
                response_parts.append(f"Date: {date_result}\n")
            
            if time_result:
                response_parts.append(f"Time: {time_result}\n")
        
        # If we have no specific content, provide a generic response
        if not response_parts:
            response_parts.append("I processed your request and here's what I found:\n")
            for call_id, result in function_results.items():
                function_name = call_id.split('-')[0] if '-' in call_id else call_id
                response_parts.append(f"{function_name}: {result}\n")
        
        return ''.join(response_parts)
        
    async def _run_with_crewai(self, query: str) -> str:
        """
        Runs a task using CrewAI.
        """
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
        # Dynamically build tools list from the function registry
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
            },
            {
                "type": "function",
                "function": {
                    "name": "GetWeatherData",
                    "description": "Returns weather data for the given coordinates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coordinates": {
                                "type": "array",
                                "description": "Latitude and longitude coordinates as [lat, lon]",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["coordinates"]
                    }
                }
            },
            # New functions automatically added from registry would go here
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
            
            # Keep track of any location data for potential chaining
            location_data = {}
            
            # Process all tool calls in this response
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                if function_name in function_registry:
                    # Extract arguments from the function call
                    function_args = {}
                    if tool_call.function.arguments and tool_call.function.arguments.strip() != "{}":
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            pass
                    
                    # Call the function with appropriate arguments based on function name
                    if function_name == "GetDate" or function_name == "GetTime":
                        # No arguments needed
                        function_result = function_registry[function_name]()
                    elif function_name == "GetCoordinatesOfCity":
                        # Extract city name from arguments
                        city_name = function_args.get("city_name", query)  # Default to using the query itself
                        function_result = function_registry[function_name](city_name)
                        
                        # Store coordinates for potential chaining with weather
                        if isinstance(function_result, tuple) and len(function_result) == 2:
                            location_data['coordinates'] = function_result
                            location_data['city'] = city_name
                    elif function_name == "GetWeatherData":
                        # Coordinates should be provided in the arguments
                        # If not available, try to use coordinates from previous calls
                        coordinates = function_args.get("coordinates", location_data.get('coordinates'))
                        if coordinates:
                            function_result = function_registry[function_name](coordinates)
                        else:
                            function_result = {"error": "No coordinates provided for weather lookup"}
                    else:
                        # For other functions, pass the whole arguments dict
                        function_result = function_registry[function_name]()
                    
                    # Add the function result to messages
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id, 
                        "content": str(function_result)
                    })
            
            # Check if we need additional function calls based on context
            # For example, if we have location data but no weather data was requested
            if ('coordinates' in location_data and 
                not any(call.function.name == "GetWeatherData" for call in response.choices[0].message.tool_calls) and
                any(word in query.lower() for word in ["weather", "temperature", "climate", "conditions"])):
                
                # The query mentioned weather but the model didn't call the weather function
                # Explicitly add a follow-up message to ask about weather
                follow_up_message = {
                    "role": "user",
                    "content": f"Please also tell me about the current weather conditions in {location_data.get('city', 'the location')}"
                }
                messages.append(follow_up_message)
            
            # Debug function results
            print("\n[DEBUG] Function results:")
            
            # Store function results for potential fallback response
            function_results = {}
            function_names = {}
            
            # Track each function call and its result
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                function_names[tool_call_id] = function_name
                
                # Find the corresponding result message
                result = None
                for msg in messages:
                    if msg.get('role') == 'tool' and msg.get('tool_call_id') == tool_call_id:
                        result = msg.get('content')
                        break
                        
                function_results[tool_call_id] = result
                print(f"  Function: {function_name}, Result: {result}")
            
            # Send the function results back for a final response
            try:
                final_response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    tools=tools
                )
                
                response_content = final_response.choices[0].message.content
                
                # Check if we got a valid response
                if not response_content or response_content.strip() == "None":
                    # Generate a fallback response using function results
                    fallback_response = self._generate_fallback_response(query, function_results, location_data)
                    return fallback_response
                
                return response_content
            except Exception as e:
                print(f"Error getting final response: {str(e)}")
                # Generate a fallback response using function results
                fallback_response = self._generate_fallback_response(query, function_results, location_data)
                return fallback_response
        else:
            # Return direct response if no function calls
            return response.choices[0].message.content
