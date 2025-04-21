"""Comprehensive test for Agent4o framework with enhanced capabilities.

This test script demonstrates the various capabilities of the Agent4o framework,
including date/time retrieval, coordinate lookup, weather data, and comparison features.

Author: Mansoor Ali
Date: April 2025
"""

import asyncio
import json
from pprint import pprint
from datetime import datetime
from Agent4o import ReActAgent, configure_azure_openai

async def test_Agent4o():
    """Run comprehensive tests on the Agent4o framework."""
    print("=" * 80)
    print("AGENT4O COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("\nStarting test at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Configure Azure OpenAI and get the model name
    print("\nConfiguring Azure OpenAI...")
    model_name = await configure_azure_openai()
    print(f"Using model: {model_name}")
    
    # Create Agent4o instance with direct function calling
    print("\nCreating Agent4o instance...")
    agent = await ReActAgent.create(
        name="TestAgent",
        role="AI Assistant",
        goal="Assist users with intelligent responses using available functions",
        tools=[],  # No CrewAI tools needed for these tests
        llm_model_name=model_name,
        use_direct_calling=True
    )
    
    # Define test categories and their queries
    test_categories = {
        "Basic Functionality": [
            "What is the date today?",
            "What time is it now?"
        ],
        "Location Services": [
            "What are the coordinates of New York City?",
            "Where is Tokyo located?"
        ],
        "Weather Information": [
            "What's the weather like in London?",
            "Tell me about the weather in Paris right now",
            "How hot is it in Tokyo today?"
        ],
        "Multi-function Queries": [
            "What is the current date, time and weather in Berlin?",
            "Tell me the time and temperature in Sydney"
        ],
        "Comparison Features": [
            "Compare the weather in London and Paris",
            "Which is warmer right now, New York or Tokyo?"
        ]
    }
    
    # Process all tests by category
    all_results = []
    
    for category, queries in test_categories.items():
        print("\n" + "=" * 80)
        print(f"TEST CATEGORY: {category}")
        print("=" * 80)
        
        category_results = []
        for i, query in enumerate(queries):
            print(f"\nTest {i+1}: {query}")
            print("-" * 50)
            
            try:
                # Process the query
                start_time = datetime.now()
                result = await agent.run_task(query)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Display the result
                print(f"\nAgent4o Response (completed in {execution_time:.2f}s):")
                print("---")
                print(result)
                print("---")
                
                # Store the result
                category_results.append({
                    "query": query,
                    "result": result,
                    "execution_time": execution_time
                })
                
            except Exception as e:
                print(f"Error testing query '{query}': {str(e)}")
                category_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        all_results.append({
            "category": category,
            "tests": category_results
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = sum(len(category["tests"]) for category in all_results)
    error_count = sum(1 for category in all_results for test in category["tests"] if "error" in test)
    success_rate = ((total_tests - error_count) / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\nTotal tests run: {total_tests}")
    print(f"Successful tests: {total_tests - error_count}")
    print(f"Tests with errors: {error_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Calculate average execution time for successful tests
    execution_times = [test["execution_time"] for category in all_results 
                      for test in category["tests"] if "execution_time" in test]
    
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print(f"Average execution time: {avg_time:.2f} seconds")
    
    print("\nTest completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    return all_results

def display_test_highlights(results):
    """Display highlighted results from each test category."""
    print("\n" + "=" * 80)
    print("TEST HIGHLIGHTS")
    print("=" * 80)
    
    for category in results:
        print(f"\n{category['category']}:")
        print("-" * 50)
        
        # Display one successful result from each category if available
        success_examples = [test for test in category["tests"] if "error" not in test]
        if success_examples:
            example = success_examples[0]
            print(f"Query: {example['query']}")
            print(f"Response: {example['result'][:150]}..." if len(example['result']) > 150 else example['result'])
            print(f"Time: {example['execution_time']:.2f}s")
        else:
            print("No successful tests in this category")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    # Run the test suite
    all_results = asyncio.run(test_Agent4o())
    
    # Display highlights from the tests
    display_test_highlights(all_results)
