"""
Test script to specifically test the booking functionality with the updated API format.
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.bot.chatbot import CalendarAgent
from src.api.cal_api import CalAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_direct_booking():
    """Test booking directly using the Cal API client."""
    logger.info("Testing direct booking with Cal API client")
    
    async with CalAPIClient() as cal_api:
        # Test API connection
        connection_result = await cal_api.test_api_connection()
        logger.info(f"API connection test result: {connection_result}")
        
        if not connection_result:
            logger.error("Failed to connect to Cal.com API")
            return
        
        # Get user inputs for booking
        email = input("Enter your email: ")
        name = input("Enter your name: ")
        
        # Use the exact date and time from the working curl command
        use_exact_curl_params = input("Use the exact parameters from working curl command? (y/n): ").lower() == 'y'
        
        if use_exact_curl_params:
            # Use the exact same parameters that worked in the curl command
            date_str = "2025-05-27"
            time_str = "22:30"
            start_time = "2025-05-27T22:30:00.000Z"
            logger.info("Using exact date and time from the working curl command")
        else:
            # Calculate date for tomorrow
            tomorrow = datetime.now() + timedelta(days=1)
            date_str = tomorrow.strftime("%Y-%m-%d")
            
            # Let the user choose a time or use a default
            time_str = input(f"Enter time (HH:MM) for {date_str} or press Enter for 14:30: ")
            if not time_str:
                time_str = "14:30"
            
            # Format the start time
            start_time = f"{date_str}T{time_str}:00.000Z"
        
        logger.info(f"Making booking request for {start_time}")
        reason = input("Enter reason for meeting (or press Enter for default): ") or "Testing Cal.com API"
        
        # Use the known working event type ID
        event_type_id = "2457598"
        
        # Make the booking request
        booking_result = await cal_api.book_event(
            event_type_id=event_type_id,
            start_time=start_time,
            name=name,
            email=email,
            reason=reason
        )
        
        # Display the result
        logger.info(f"Booking result: {json.dumps(booking_result, indent=2)}")
        
        if booking_result.get("status") == "success":
            print("\n✅ Booking successful!")
            booking_data = booking_result.get("booking", {})
            print(f"Booking ID: {booking_data.get('uid', 'Unknown')}")
            print(f"Start time: {booking_data.get('startTime', 'Unknown')}")
            print(f"End time: {booking_data.get('endTime', 'Unknown')}")
        else:
            print("\n❌ Booking failed!")
            print(f"Error message: {booking_result.get('message', 'Unknown error')}")

async def test_chatbot_booking():
    """Test booking through the chatbot interface."""
    logger.info("Testing booking through chatbot")
    
    # Create the agent
    agent = CalendarAgent()
    await agent.initialize()
    
    try:
        # Get user email
        user_email = input("Enter your email: ")
        
        # Process a booking request
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = tomorrow.strftime("%Y-%m-%d")
        time_str = "14:30"
        
        booking_message = f"I want to book a meeting for {date_str} at {time_str}"
        print(f"\nSending message: '{booking_message}'")
        
        # Process the message
        response = await agent.process_message(booking_message, user_email)
        
        # Print the assistant's response
        print(f"\nBot: {response['response']}")
        print(f"Action taken: {response['action_taken']}")
        
        # Print detailed result
        print("\nDetailed result:")
        if response.get('details'):
            print(json.dumps(response.get('details'), indent=2))
    
    finally:
        # Clean up resources
        await agent.cleanup()
        logger.info("Chatbot test completed")

async def main():
    """Run the test."""
    print("=== Cal.com Booking Test ===\n")
    
    # Choose test type
    print("Choose a test:")
    print("1. Test direct API booking")
    print("2. Test chatbot booking")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == '1':
        await test_direct_booking()
    elif choice == '2':
        await test_chatbot_booking()
    else:
        print("Invalid choice. Exiting.")
    
    print("\n=== Test Complete ===")

# Run the script
if __name__ == "__main__":
    asyncio.run(main()) 