import pytest
from unittest.mock import patch, AsyncMock
import json
from datetime import datetime, timedelta

from src.bot.chatbot import CalendarAgent
from src.bot.openai_integration import OpenAIFunctionCaller

# Mock data for tests
MOCK_EVENT_DATA = {
    "bookings": [
        {
            "uid": "booking123",
            "title": "Project Review Meeting",
            "startTime": "2023-06-15T14:00:00.000Z",
            "endTime": "2023-06-15T14:30:00.000Z",
            "eventType": {"length": 30},
            "organizer": {"name": "John Doe"},
            "attendees": [{"name": "User", "email": "user@example.com"}]
        },
        {
            "uid": "booking456",
            "title": "Sprint Planning",
            "startTime": "2023-06-16T10:00:00.000Z",
            "endTime": "2023-06-16T11:00:00.000Z",
            "eventType": {"length": 60},
            "organizer": {"name": "Jane Smith"},
            "attendees": [{"name": "User", "email": "user@example.com"}]
        }
    ]
}

MOCK_CANCEL_RESULT = {
    "action": "cancel_event",
    "status": "success",
    "event_id": "booking123",
    "message": "Event cancelled successfully"
}

MOCK_CANCEL_ERROR = {
    "action": "cancel_event",
    "status": "error",
    "event_id": "invalid-id",
    "message": "Failed to cancel event: HTTP error 404: Not found"
}

MOCK_API_ERROR = {
    "action": "cancel_event",
    "status": "error",
    "event_id": "booking123",
    "message": "Failed to cancel event: HTTP error 500: Database error occurred"
}

@pytest.mark.asyncio
async def test_cancel_event_direct_with_id():
    """Test cancellation flow with a direct event ID."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller') as MockOpenAI, \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
         
        # Setup mocks
        mock_function_caller = AsyncMock()
        mock_function_caller.process_with_function_calling.return_value = (
            "I've successfully cancelled your event with ID booking123.",
            {
                "action": "cancel_event",
                "status": "success",
                "detected_intent": "cancellation_intent",
                "event_id": "booking123"
            }
        )
        MockOpenAI.return_value = mock_function_caller
        
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.cancel_booking.return_value = {
            "status": "success",
            "booking_id": "booking123",
            "details": {}
        }
        MockCalAPI.return_value = mock_cal_api
        
        # Create agent and process message
        agent = CalendarAgent()
        await agent.initialize()
        
        # Process message
        result = await agent.process_message("I want to cancel my event with ID booking123", "user@example.com")
        
        # Check results
        assert result["action_taken"] == "cancel_event"
        assert "successfully cancelled" in result["response"].lower()

@pytest.mark.asyncio
async def test_cancel_event_with_date_time():
    """Test cancellation flow with date and time instead of ID."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller') as MockOpenAI, \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
         
        # Setup mocks
        mock_function_caller = AsyncMock()
        
        # First response shows available events
        mock_function_caller.process_with_function_calling.side_effect = [
            (
                "I found these events for 2023-06-15:\n- Project Review Meeting at 14:00 (ID: booking123)\n\nWould you like me to cancel this event? Please confirm.",
                {
                    "action": "cancel_confirmation_needed",
                    "status": "pending",
                    "detected_intent": "cancellation_intent",
                    "event_id": "booking123"
                }
            ),
            (
                "I've successfully cancelled your event with ID booking123.",
                {
                    "action": "cancel_event",
                    "status": "success",
                    "detected_intent": "confirmation_intent",
                    "event_id": "booking123"
                }
            )
        ]
        MockOpenAI.return_value = mock_function_caller
        
        # Setup Cal API mocks
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.list_bookings.return_value = {
            "status": "success",
            "bookings": MOCK_EVENT_DATA["bookings"]
        }
        mock_cal_api.cancel_booking.return_value = {
            "status": "success",
            "booking_id": "booking123",
            "details": {}
        }
        MockCalAPI.return_value = mock_cal_api
        
        # Create agent and process messages
        agent = CalendarAgent()
        await agent.initialize()
        
        # First message asks to cancel by date
        result1 = await agent.process_message("Cancel my meeting on June 15th", "user@example.com")
        
        # Check first response (confirmation needed)
        assert result1["action_taken"] == "cancel_confirmation_needed"
        assert "would you like me to cancel" in result1["response"].lower()
        
        # Second message confirms cancellation
        result2 = await agent.process_message("Yes, please cancel it", "user@example.com")
        
        # Check second response (cancellation done)
        assert result2["action_taken"] == "cancel_event"
        assert "successfully cancelled" in result2["response"].lower()

@pytest.mark.asyncio
async def test_cancel_nonexistent_event():
    """Test cancellation of a non-existent event."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller') as MockOpenAI, \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
         
        # Setup mocks
        mock_function_caller = AsyncMock()
        mock_function_caller.process_with_function_calling.return_value = (
            "I couldn't find an event with ID invalid-id. Please check the ID and try again.",
            {
                "action": "cancel_event",
                "status": "error",
                "detected_intent": "cancellation_intent",
                "message": "HTTP error 404: Not found",
                "event_id": "invalid-id"
            }
        )
        MockOpenAI.return_value = mock_function_caller
        
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.cancel_booking.return_value = {
            "status": "error",
            "message": "HTTP error 404: Not found",
            "booking_id": "invalid-id"
        }
        MockCalAPI.return_value = mock_cal_api
        
        # Create agent and process message
        agent = CalendarAgent()
        await agent.initialize()
        
        result = await agent.process_message("Cancel event with ID invalid-id", "user@example.com")
        
        # Check results
        assert result["action_taken"] == "cancel_event"
        assert result["details"]["status"] == "error"
        assert "couldn't find" in result["response"].lower()

@pytest.mark.asyncio
async def test_cancel_event_api_error():
    """Test handling of API errors during cancellation."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller') as MockOpenAI, \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
         
        # Setup mocks
        mock_function_caller = AsyncMock()
        mock_function_caller.process_with_function_calling.return_value = (
            "I'm sorry, but the calendar service is currently experiencing technical difficulties. Please try again later.",
            {
                "action": "cancel_event",
                "status": "error",
                "detected_intent": "cancellation_intent",
                "message": "HTTP error 500: Database error occurred",
                "event_id": "booking123"
            }
        )
        MockOpenAI.return_value = mock_function_caller
        
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.cancel_booking.return_value = {
            "status": "error",
            "message": "HTTP error 500: Database error occurred",
            "booking_id": "booking123"
        }
        MockCalAPI.return_value = mock_cal_api
        
        # Create agent and process message
        agent = CalendarAgent()
        await agent.initialize()
        
        result = await agent.process_message("Cancel my meeting with ID booking123", "user@example.com")
        
        # Check results
        assert result["action_taken"] == "cancel_event"
        assert result["details"]["status"] == "error"
        assert "technical difficulties" in result["response"].lower()

@pytest.mark.asyncio
async def test_cancel_no_email_provided():
    """Test cancellation flow when no email is provided."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller') as MockOpenAI, \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
         
        # Setup mocks
        mock_function_caller = AsyncMock()
        mock_function_caller.process_with_function_calling.return_value = (
            "I need your email address to look up your events. Please provide your email.",
            {
                "action": "email_needed",
                "status": "pending",
                "detected_intent": "cancellation_intent"
            }
        )
        MockOpenAI.return_value = mock_function_caller
        
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        MockCalAPI.return_value = mock_cal_api
        
        # Create agent and process message (without email)
        agent = CalendarAgent()
        await agent.initialize()
        
        result = await agent.process_message("I want to cancel my meeting tomorrow")
        
        # Check results
        assert result["action_taken"] == "email_needed"
        assert "need your email" in result["response"].lower() 