import os
import requests

# Use environment variables for configuration
API_BASE_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("CHATBOT_API_KEY", "test-api-key")

HEADERS = {"X-API-Key": API_KEY}


def send_message(message: str, user_email: str, conversation_id: str = None):
    """
    Send a message to the chatbot backend and return the response.
    """
    url = f"{API_BASE_URL}/chat"
    payload = {
        "message": message,
        "user_email": user_email,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_conversation_messages(conversation_id: str):
    """
    Retrieve the message history for a conversation.
    """
    url = f"{API_BASE_URL}/api/conversations/{conversation_id}/messages"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)} 