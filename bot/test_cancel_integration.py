"""
Integration test script for the cancellation flow.
This script demonstrates the full cancellation flow from a user perspective.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.bot.chatbot import CalendarAgent

async def main():
    """Demonstrate the cancellation flow with a simulated conversation."""
    print("=== Calendar Agent Cancellation Flow Test ===\n")
    
    # Create the agent
    agent = CalendarAgent()
    await agent.initialize()
    
    # User's email for this test
    user_email = "test@example.com"
    
    # Simulate a conversation with the agent
    messages = [
        "Hello, I need to cancel a meeting",
        "I want to cancel my appointment on June 15th",
        "Yes, please cancel it",
        "Thanks for your help!"
    ]
    
    # Process each message
    for i, message in enumerate(messages):
        print(f"USER: {message}")
        
        # Process the message
        response = await agent.process_message(message, user_email)
        
        # Print the assistant's response
        print(f"ASSISTANT: {response['response']}")
        print(f"ACTION: {response['action_taken']}")
        
        if response.get('details', {}).get('events'):
            # Display events if they were returned
            print("\nEvents:")
            for event in response['details']['events']:
                print(f"- {event.get('date')} at {event.get('time')}: {event.get('title')} (ID: {event.get('id')})")
        
        # Add a delay for readability
        await asyncio.sleep(0.5)
        print("\n" + "-"*50 + "\n")
    
    # Clean up resources
    await agent.cleanup()
    print("=== Test Complete ===")

# This is a script that can be run directly
if __name__ == "__main__":
    asyncio.run(main()) 