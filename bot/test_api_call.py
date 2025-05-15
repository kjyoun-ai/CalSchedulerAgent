"""
Test script to make a real API call with the chatbot.

This script tests the Cal.com API integration with the chatbot by sending a message 
and processing the response, showing the full interaction including API calls.
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.bot.chatbot import CalendarAgent
from src.api.cal_api import CalAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_booking_api():
    """Test the booking API call."""
    logger.info("Testing Cal.com booking API...")
    
    async with CalAPIClient() as cal_api:
        # Test API connection
        connection_result = await cal_api.test_api_connection()
        logger.info(f"API connection test result: {connection_result}")
        
        if not connection_result:
            logger.error("Failed to connect to Cal.com API")
            return
        
        # List available events
        user_email = input("Enter your email to list events: ")
        
        bookings_result = await cal_api.list_bookings(user_email)
        logger.info(f"Bookings result status: {bookings_result.get('status')}")
        
        if bookings_result.get('status') == 'success' and bookings_result.get('bookings'):
            bookings = bookings_result.get('bookings', [])
            logger.info(f"Found {len(bookings)} bookings")
            
            print("\nYour bookings:")
            for i, booking in enumerate(bookings):
                start_time = booking.get('startTime', '').split('T')
                date = start_time[0] if len(start_time) > 0 else 'Unknown'
                time = start_time[1].split('.')[0] if len(start_time) > 1 else 'Unknown'
                print(f"{i+1}. Date: {date}, Time: {time}, ID: {booking.get('uid', 'Unknown')}")
                print(f"   Title: {booking.get('title', 'Untitled')}")
                print()
            
            if bookings:
                booking_id = input("Enter booking ID to cancel (or press Enter to skip): ")
                if booking_id:
                    # Confirm cancellation
                    confirm = input(f"Are you sure you want to cancel booking {booking_id}? (y/n): ")
                    if confirm.lower() == 'y':
                        cancel_result = await cal_api.cancel_booking(booking_id)
                        logger.info(f"Cancellation result: {cancel_result}")
                        
                        if cancel_result.get('status') == 'success':
                            print(f"Successfully cancelled booking {booking_id}")
                        else:
                            print(f"Failed to cancel booking: {cancel_result.get('message', 'Unknown error')}")
        else:
            logger.error(f"Failed to list bookings: {bookings_result.get('message', 'Unknown error')}")

async def test_chatbot():
    """Test the chatbot with a real message."""
    logger.info("Testing chatbot...")
    
    # Create the agent
    agent = CalendarAgent()
    await agent.initialize()
    
    try:
        # Get user email
        user_email = input("Enter your email: ")
        
        # Chat loop
        print("\n=== Chat with Calendar Bot ===")
        print("Type 'exit' or 'quit' to end the conversation")
        
        while True:
            # Get user message
            user_message = input("\nYou: ")
            if user_message.lower() in ['exit', 'quit']:
                break
            
            # Process the message
            response = await agent.process_message(user_message, user_email)
            
            # Print the assistant's response
            print(f"\nBot: {response['response']}")
            print(f"[Debug] Action: {response['action_taken']}")
            
            # Print additional details for debugging
            if 'details' in response and response['details']:
                debug_info = {
                    'status': response['details'].get('status'),
                    'action': response['details'].get('action')
                }
                
                # Add events to debug info if present
                if 'events' in response['details']:
                    events_count = len(response['details']['events'])
                    debug_info['events_count'] = events_count
                    
                    if events_count > 0:
                        print("\nEvents found:")
                        for event in response['details']['events']:
                            print(f"- {event.get('date')} at {event.get('time')}: {event.get('title')} (ID: {event.get('id')})")
                
                print(f"[Debug] Details: {json.dumps(debug_info)}")
    
    finally:
        # Clean up resources
        await agent.cleanup()
        logger.info("Chatbot test completed")

async def main():
    """Run the test."""
    print("=== Cal.com API and Chatbot Test ===\n")
    
    # Choose test type
    print("Choose a test:")
    print("1. Test direct API calls (list/cancel bookings)")
    print("2. Test chatbot with conversation")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == '1':
        await test_booking_api()
    elif choice == '2':
        await test_chatbot()
    else:
        print("Invalid choice. Exiting.")
    
    print("\n=== Test Complete ===")

# Run the script
if __name__ == "__main__":
    asyncio.run(main()) 