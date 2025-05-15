"""
Integrated booking flow test script.

This script tests the entire booking flow from user message to API call with our fixed implementation.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.bot.chatbot import CalendarAgent

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_integrated_booking():
    """Test the integrated booking flow from user message to API call."""
    logger.info("=== Integrated Booking Flow Test ===")
    
    # Create the chatbot
    agent = CalendarAgent()
    await agent.initialize()
    
    # Add user email to the conversation context
    agent.conversation_context["current_user_email"] = "test@example.com"
    
    # Get tomorrow's date for testing
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    
    # Format time in the evening (8:30 PM) when slots are likely available
    time_str = "20:30"
    
    logger.info(f"Testing booking for {tomorrow_str} at {time_str}")
    
    # Method 1: Test the book_meeting method directly
    logger.info("1. Testing direct booking method...")
    booking_result = await agent.book_meeting(
        date=tomorrow_str,
        time=time_str,
        name="Test User",
        reason="Testing the booking flow"
    )
    
    logger.info(f"Direct booking result: {booking_result}")
    
    # Method 2: Test the chatbot with a booking intent message
    logger.info("\n2. Testing booking through chatbot message processing...")
    message = f"Can you book a meeting for me tomorrow at 8:30pm? It's for discussing project requirements."
    
    logger.info(f"User message: {message}")
    response = await agent.process_message(message)
    
    logger.info(f"Chatbot response: {response}")
    
    # Method 3: Test the chatbot with explicit date/time
    logger.info("\n3. Testing booking with explicit date/time...")
    message = f"Book a meeting on {tomorrow_str} at {time_str} to discuss integration testing"
    
    logger.info(f"User message: {message}")
    response = await agent.process_message(message)
    
    logger.info(f"Chatbot response: {response}")
    
    logger.info("=== Integrated Booking Flow Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_integrated_booking()) 