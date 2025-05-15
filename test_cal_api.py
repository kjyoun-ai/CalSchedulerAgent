"""
Tests for the Cal.com API client.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.api.cal_api import CalAPIClient
import httpx

# Sample successful responses for mocking
MOCK_EVENT_TYPES_RESPONSE = {
    "event_types": [
        {
            "id": "1",
            "title": "30 Minute Meeting",
            "description": "A quick 30 minute meeting",
            "length": 30
        },
        {
            "id": "2",
            "title": "60 Minute Meeting",
            "description": "A standard 60 minute meeting",
            "length": 60
        }
    ]
}

MOCK_SLOTS_RESPONSE = {
    "slots": {
        "2023-06-01": [
            "2023-06-01T09:00:00.000Z",
            "2023-06-01T10:00:00.000Z",
            "2023-06-01T11:00:00.000Z"
        ]
    }
}

MOCK_BOOKING_RESPONSE = {
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

MOCK_BOOKINGS_RESPONSE = {
    "bookings": [
        {
            "uid": "booking1",
            "attendees": [
                {"name": "John Doe", "email": "john@example.com"}
            ],
            "title": "Meeting 1"
        },
        {
            "uid": "booking2",
            "attendees": [
                {"name": "Jane Smith", "email": "jane@example.com"}
            ],
            "title": "Meeting 2"
        },
        {
            "uid": "booking3",
            "attendees": [
                {"name": "John Doe", "email": "john@example.com"}
            ],
            "title": "Meeting 3"
        }
    ]
}

MOCK_CANCEL_RESPONSE = {"message": "Booking cancelled"}

MOCK_RESCHEDULE_RESPONSE = {"message": "Booking rescheduled"}

@pytest.mark.asyncio
async def test_test_api_connection():
    """Test the test_api_connection method."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        cal_client = CalAPIClient()
        result = await cal_client.test_api_connection()
        
        assert result is True
        mock_get.assert_called_once()
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_get_event_types():
    """Test the get_event_types method."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_EVENT_TYPES_RESPONSE
        mock_get.return_value = mock_response
        
        cal_client = CalAPIClient()
        result = await cal_client.get_event_types()
        
        assert result == MOCK_EVENT_TYPES_RESPONSE
        mock_get.assert_called_once()
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_get_available_slots():
    """Test the get_available_slots method."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_SLOTS_RESPONSE
        mock_get.return_value = mock_response
        
        cal_client = CalAPIClient()
        result = await cal_client.get_available_slots(
            event_type_id="1",
            start_time="2023-06-01T00:00:00Z",
            end_time="2023-06-02T23:59:59Z"
        )
        
        assert result == MOCK_SLOTS_RESPONSE
        mock_get.assert_called_once()
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_book_event():
    """Test the book_event method."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.post') as mock_post:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_BOOKING_RESPONSE
        mock_post.return_value = mock_response
        
        cal_client = CalAPIClient()
        result = await cal_client.book_event(
            event_type_id="1",
            start_time="2023-06-01T09:00:00.000Z",
            name="John Doe",
            email="john@example.com",
            reason="Discuss project"
        )
        
        assert result["status"] == "success"
        assert result["booking"] == MOCK_BOOKING_RESPONSE
        
        # Verify the payload sent to the API
        called_kwargs = mock_post.call_args.kwargs
        assert "json" in called_kwargs
        
        payload = called_kwargs["json"]
        assert payload["eventTypeId"] == "1"
        assert payload["start"] == "2023-06-01T09:00:00.000Z"
        assert len(payload["attendees"]) == 1
        assert payload["attendees"][0]["name"] == "John Doe"
        assert payload["attendees"][0]["email"] == "john@example.com"
        assert "responses" in payload
        assert payload["responses"]["reason"] == "Discuss project"
        
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_book_event_error():
    """Test the book_event method with an error response."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.post') as mock_post:
        
        # Simulate an HTTP error
        mock_post.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(
                status_code=400,
                text="Invalid request",
            )
        )
        
        cal_client = CalAPIClient()
        result = await cal_client.book_event(
            event_type_id="1",
            start_time="invalid_time",
            name="John Doe",
            email="john@example.com"
        )
        
        assert result["status"] == "error"
        assert "HTTP error 400" in result["message"]
        
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_list_bookings_all():
    """Test listing all bookings without email filter."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_BOOKINGS_RESPONSE
        mock_get.return_value = mock_response
        cal_client = CalAPIClient()
        result = await cal_client.list_bookings()
        assert result["status"] == "success"
        assert len(result["bookings"]) == 3
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_list_bookings_filter_email():
    """Test listing bookings filtered by attendee email."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_BOOKINGS_RESPONSE
        mock_get.return_value = mock_response
        cal_client = CalAPIClient()
        result = await cal_client.list_bookings(email="john@example.com")
        assert result["status"] == "success"
        assert len(result["bookings"]) == 2
        for booking in result["bookings"]:
            assert any(a["email"] == "john@example.com" for a in booking["attendees"])
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_list_bookings_error():
    """Test error handling in list_bookings."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.get') as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="Internal Server Error")
        )
        cal_client = CalAPIClient()
        result = await cal_client.list_bookings()
        assert result["status"] == "error"
        assert "HTTP error 500" in result["message"]
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_cancel_booking_success():
    """Test successful cancellation of a booking."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.delete') as mock_delete:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"message": "Booking cancelled"}'
        mock_response.json.return_value = MOCK_CANCEL_RESPONSE
        mock_delete.return_value = mock_response
        cal_client = CalAPIClient()
        result = await cal_client.cancel_booking(booking_id="booking1")
        assert result["status"] == "success"
        assert result["booking_id"] == "booking1"
        assert result["details"] == MOCK_CANCEL_RESPONSE
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_cancel_booking_error():
    """Test error handling in cancel_booking."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.delete') as mock_delete:
        mock_delete.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(status_code=404, text="Not Found")
        )
        cal_client = CalAPIClient()
        result = await cal_client.cancel_booking(booking_id="booking1")
        assert result["status"] == "error"
        assert result["booking_id"] == "booking1"
        assert "HTTP error 404" in result["message"]
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_reschedule_booking_success():
    """Test successful rescheduling of a booking."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.patch') as mock_patch:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"message": "Booking rescheduled"}'
        mock_response.json.return_value = MOCK_RESCHEDULE_RESPONSE
        mock_patch.return_value = mock_response
        cal_client = CalAPIClient()
        result = await cal_client.reschedule_booking(booking_id="booking1", new_start_time="2024-06-21T10:00:00.000Z")
        assert result["status"] == "success"
        assert result["booking_id"] == "booking1"
        assert result["details"] == MOCK_RESCHEDULE_RESPONSE
        await cal_client.client.aclose()

@pytest.mark.asyncio
async def test_reschedule_booking_error():
    """Test error handling in reschedule_booking."""
    with patch('src.api.cal_api.CAL_API_KEY', 'test_api_key'), \
         patch('src.api.cal_api.CAL_API_URL', 'https://api.cal.com/v1'), \
         patch('httpx.AsyncClient.patch') as mock_patch:
        mock_patch.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(status_code=400, text="Bad Request")
        )
        cal_client = CalAPIClient()
        result = await cal_client.reschedule_booking(booking_id="booking1", new_start_time="2024-06-21T10:00:00.000Z")
        assert result["status"] == "error"
        assert result["booking_id"] == "booking1"
        assert "HTTP error 400" in result["message"]
        await cal_client.client.aclose() 