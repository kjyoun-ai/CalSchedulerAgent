"""
Configuration settings for the Cal.com Scheduler Agent.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CAL_API_KEY = os.environ.get("CAL_API_KEY", "")
CAL_API_URL = os.environ.get("CAL_API_URL", "https://api.cal.com/v1")

# Model Configuration
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# Application Settings
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))

def validate_config() -> bool:
    """
    Validate that all required configuration variables are set.
    Returns True if all required configs are present, False otherwise.
    """
    missing_vars = []
    
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    
    if not CAL_API_KEY:
        missing_vars.append("CAL_API_KEY")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def get_api_key(key_name: str) -> Optional[str]:
    """Get an API key from environment variables."""
    key = os.environ.get(key_name, "")
    if not key:
        logger.warning(f"{key_name} is not set in environment variables")
    return key 