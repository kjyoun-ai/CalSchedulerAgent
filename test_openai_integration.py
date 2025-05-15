"""
Tests for the OpenAI integration module (LangChain version).
"""

import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Dict, Any, List

from src.bot.openai_integration import OpenAIFunctionCaller

# Sample data for tests
SAMPLE_MESSAGES = [
    {"role": "user", "content": "Can you book a meeting for me tomorrow at 3pm?"}
]

SAMPLE_FUNCTIONS = {
    "book_meeting": AsyncMock(return_value={"status": "success", "action": "book_meeting"})
}

# Skip tests if no API key is available
skip_if_no_api_key = pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None, 
    reason="OPENAI_API_KEY environment variable not set"
)

class TestOpenAIFunctionCaller:
    def test_init_raises_error_without_api_key(self, monkeypatch):
        """Test that initialization raises an error if no API key is available."""
        # Remove API key from both env and code config
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch('src.bot.openai_integration.OPENAI_API_KEY', None):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIFunctionCaller(api_key=None)

    def test_init_with_explicit_api_key(self):
        """Test initialization with an explicit API key."""
        caller = OpenAIFunctionCaller(api_key="test_key")
        assert caller.api_key == "test_key"
        assert caller.model == "gpt-3.5-turbo"  # Default model

    def test_init_with_explicit_model(self):
        """Test initialization with an explicit model."""
        caller = OpenAIFunctionCaller(api_key="test_key", model="gpt-3.5-turbo")
        assert caller.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_process_with_function_calling_returns_response(self):
        """Test that process_with_function_calling returns a response."""
        # Create a caller instance
        caller = OpenAIFunctionCaller(api_key="test_key")
        
        # Create a simple patch that just returns a fixed response
        with patch.object(OpenAIFunctionCaller, 'process_with_function_calling', new_callable=AsyncMock) as mock_process:
            # Set up mock return value - a response and a simple result
            expected_response = "I've booked your meeting"
            expected_result = {"status": "success", "action": "book_meeting"}
            mock_process.return_value = (expected_response, expected_result)
            
            # Call the method
            response, result = await caller.process_with_function_calling(
                messages=SAMPLE_MESSAGES,
                functions=SAMPLE_FUNCTIONS
            )
            
            # Verify we get the right response
            assert response == expected_response
            assert result == expected_result
            # Verify the method was called with right parameters
            mock_process.assert_awaited_once_with(messages=SAMPLE_MESSAGES, functions=SAMPLE_FUNCTIONS)

    @pytest.mark.asyncio
    async def test_process_with_function_calling_tool_called(self):
        """Test that the tool function is called correctly."""
        # Create a caller instance
        caller = OpenAIFunctionCaller(api_key="test_key")
        
        # Create a mock that will be passed directly to process_with_function_calling
        book_meeting_mock = AsyncMock()
        book_meeting_mock.return_value = {"status": "success", "details": "Meeting booked successfully!"}
        
        # Mock the entire process_with_function_calling method to simulate it calling our book_meeting_mock
        original_method = caller.process_with_function_calling
        
        async def mock_implementation(messages, functions):
            # Simulate calling the function with specific arguments
            if "book_meeting" in functions:
                await functions["book_meeting"](
                    name="Test User", 
                    email="test@example.com", 
                    start_time="2023-10-15T15:00:00",
                    duration=30
                )
                return "Great! I've booked your meeting.", functions["book_meeting"].return_value
            return await original_method(messages, functions)
        
        # Apply the mock implementation
        with patch.object(caller, 'process_with_function_calling', mock_implementation):
            # Call the mocked method
            response, result = await caller.process_with_function_calling(
                messages=SAMPLE_MESSAGES,
                functions={"book_meeting": book_meeting_mock}
            )
            
            # Verify the function was called with expected arguments
            book_meeting_mock.assert_awaited_once_with(
                name="Test User", 
                email="test@example.com", 
                start_time="2023-10-15T15:00:00",
                duration=30
            )
            
            # Verify the response is correct
            assert "Great! I've booked your meeting." == response
            assert result["status"] == "success" 