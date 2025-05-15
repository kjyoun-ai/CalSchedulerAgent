"""
Tests for the chatbot module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from src.bot.chatbot import CalendarAgent
import json

# Sample response for mocking
MOCK_BOOKING_RESPONSE = {
    "status": "success",
    "booking": {
        "uid": "booking123",
        "eventTypeId": "1",
        "userId": "user123",
        "startTime": "2023-06-01T09:00:00.000Z",
        "endTime": "2023-06-01T09:30:00.000Z",
        "title": "30 Minute Meeting",
        "status": "ACCEPTED",
        "attendees": [
            {
                "name": "John Doe",
                "email": "john@example.com"
            }
        ]
    }
}

MOCK_BOOKINGS_RESULT = {
    "status": "success",
    "bookings": [
        {
            "uid": "booking1",
            "attendees": [{"name": "John Doe", "email": "john@example.com"}],
            "title": "Meeting 1",
            "startTime": "2023-06-01T09:00:00.000Z",
            "eventType": {"length": 30},
            "organizer": {"name": "Alice"}
        },
        {
            "uid": "booking2",
            "attendees": [{"name": "Jane Smith", "email": "jane@example.com"}],
            "title": "Meeting 2",
            "startTime": "2023-06-02T10:00:00.000Z",
            "eventType": {"length": 60},
            "organizer": {"name": "Bob"}
        }
    ]
}

MOCK_CANCEL_RESULT = {
    "status": "success",
    "booking_id": "booking1",
    "details": {"message": "Booking cancelled"}
}

MOCK_RESCHEDULE_RESULT = {
    "status": "success",
    "booking_id": "booking1",
    "details": {"message": "Booking rescheduled"}
}

@pytest.mark.asyncio
async def test_calendar_agent_init():
    """Test CalendarAgent initialization."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'):
        agent = CalendarAgent()
        assert agent is not None
        assert agent.conversation_history == []
        assert agent.conversation_context["current_user_email"] is None

@pytest.mark.asyncio
async def test_book_meeting_success():
    """Test successful booking of a meeting."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        
        # Create an async mock for CalAPIClient
        mock_cal_api = AsyncMock()
        # Configure the mock methods with their return values
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.book_event.return_value = MOCK_BOOKING_RESPONSE
        # The mock constructor returns the mock instance
        MockCalAPI.return_value = mock_cal_api
        
        # Create CalendarAgent and set up test values
        agent = CalendarAgent()
        await agent.initialize()
        agent.conversation_context["current_user_email"] = "john@example.com"
        
        # Call the method under test
        result = await agent.book_meeting(
            date="2023-06-01",
            time="09:00",
            event_type_id="1",
            name="John Doe",
            reason="Test meeting"
        )
        
        # Verify the result
        assert result["action"] == "book_meeting"
        assert result["status"] == "success"
        assert result["booking_id"] == "booking123"
        assert result["date"] == "2023-06-01"
        assert result["time"] == "09:00"
        
        # Verify the API call
        mock_cal_api.book_event.assert_called_once_with(
            event_type_id="1",
            start_time="2023-06-01T09:00:00.000Z",
            name="John Doe",
            email="john@example.com",
            reason="Test meeting"
        )
        
        # No need to cleanup as we're using mocks

@pytest.mark.asyncio
async def test_book_meeting_no_email():
    """Test booking a meeting with no email."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        
        # Create an async mock for CalAPIClient
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        MockCalAPI.return_value = mock_cal_api
        
        # Create CalendarAgent without setting an email
        agent = CalendarAgent()
        await agent.initialize()
        
        # Call the method under test
        result = await agent.book_meeting(
            date="2023-06-01",
            time="09:00",
            name="John Doe",
            reason="Test meeting"
        )
        
        # Verify the result
        assert result["action"] == "book_meeting"
        assert result["status"] == "error"
        assert "No user email provided" in result["message"]

@pytest.mark.asyncio
async def test_book_meeting_api_error():
    """Test booking a meeting with an API error."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        
        # Create an async mock for CalAPIClient with error response
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.book_event.return_value = {
            "status": "error",
            "message": "Failed to book: Invalid time slot"
        }
        MockCalAPI.return_value = mock_cal_api
        
        # Create CalendarAgent and set up test values
        agent = CalendarAgent()
        await agent.initialize()
        agent.conversation_context["current_user_email"] = "john@example.com"
        
        # Call the method under test
        result = await agent.book_meeting(
            date="2023-06-01",
            time="09:00",
            name="John Doe",
            reason="Test meeting"
        )
        
        # Verify the result
        assert result["action"] == "book_meeting"
        assert result["status"] == "error"
        assert "time slot is not available" in result["message"]
        assert "technical_details" in result  # Check that technical details are included

@pytest.mark.asyncio
async def test_list_events_success():
    """Test successful listing of events for a user."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.list_bookings.return_value = MOCK_BOOKINGS_RESULT
        MockCalAPI.return_value = mock_cal_api
        agent = CalendarAgent()
        await agent.initialize()
        agent.conversation_context["current_user_email"] = "john@example.com"
        result = await agent.list_events()
        assert result["action"] == "list_events"
        assert result["status"] == "success"
        assert "events" in result
        assert isinstance(result["events"], list)

@pytest.mark.asyncio
async def test_list_events_no_email():
    """Test listing events with no email provided."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        MockCalAPI.return_value = mock_cal_api
        agent = CalendarAgent()
        await agent.initialize()
        # No email set
        result = await agent.list_events()
        assert result["action"] == "list_events"
        assert result["status"] == "error"
        assert "No user email provided" in result["message"]

@pytest.mark.asyncio
async def test_list_events_api_error():
    """Test listing events with an API error."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.list_bookings.return_value = {"status": "error", "message": "API failure"}
        MockCalAPI.return_value = mock_cal_api
        agent = CalendarAgent()
        await agent.initialize()
        agent.conversation_context["current_user_email"] = "john@example.com"
        result = await agent.list_events()
        assert result["action"] == "list_events"
        assert result["status"] == "error"
        assert "Failed to list events" in result["message"]
        mock_cal_api.list_bookings.assert_called_once_with("john@example.com")

@pytest.mark.asyncio
async def test_cancel_event_success():
    """Test successful cancellation of an event."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.cancel_booking.return_value = MOCK_CANCEL_RESULT
        MockCalAPI.return_value = mock_cal_api
        agent = CalendarAgent()
        await agent.initialize()
        result = await agent.cancel_event(event_id="booking1")
        assert result["action"] == "cancel_event"
        assert result["status"] == "success"
        assert result["event_id"] == "booking1"
        assert result["message"] == "Event cancelled successfully"

@pytest.mark.asyncio
async def test_cancel_event_api_error():
    """Test cancellation of an event with an API error."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.cancel_booking.return_value = {"status": "error", "message": "Not found"}
        MockCalAPI.return_value = mock_cal_api
        agent = CalendarAgent()
        await agent.initialize()
        result = await agent.cancel_event(event_id="booking1")
        assert result["action"] == "cancel_event"
        assert result["status"] == "error"
        assert "Failed to cancel event" in result["message"]
        mock_cal_api.cancel_booking.assert_called_once_with("booking1")

@pytest.mark.asyncio
async def test_reschedule_event_success():
    """Test successful rescheduling of an event."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.reschedule_booking.return_value = MOCK_RESCHEDULE_RESULT
        MockCalAPI.return_value = mock_cal_api
        agent = CalendarAgent()
        await agent.initialize()
        result = await agent.reschedule_event(event_id="booking1", new_date="2024-06-21", new_time="10:00")
        assert result["action"] == "reschedule_event"
        assert result["status"] == "success"
        assert result["event_id"] == "booking1"
        assert result["new_date"] == "2024-06-21"
        assert result["new_time"] == "10:00"

@pytest.mark.asyncio
async def test_reschedule_event_api_error():
    """Test rescheduling of an event with an API error."""
    with patch('src.bot.chatbot.OPENAI_API_KEY', 'test_api_key'), \
         patch('src.bot.chatbot.OpenAIFunctionCaller'), \
         patch('src.bot.chatbot.CalAPIClient') as MockCalAPI:
        mock_cal_api = AsyncMock()
        mock_cal_api.test_api_connection.return_value = True
        mock_cal_api.reschedule_booking.return_value = {
            "status": "error",
            "message": "Failed to reschedule: Invalid time"
        }
        MockCalAPI.return_value = mock_cal_api
        agent = CalendarAgent()
        await agent.initialize()
        result = await agent.reschedule_event(event_id="booking1", new_date="2024-06-21", new_time="10:00")
        assert result["action"] == "reschedule_event"
        assert result["status"] == "error"
        assert "Failed to reschedule event" in result["message"]
        mock_cal_api.reschedule_booking.assert_called_once_with("booking1", "2024-06-21T10:00:00.000Z") 