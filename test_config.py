"""
Tests for the configuration module.
"""

import os
import pytest
from src.utils.config import validate_config
from unittest.mock import patch

def test_validate_config_missing_vars(monkeypatch):
    """Test that validate_config returns False if required vars are missing."""
    # Temporarily clear environment variables
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CAL_API_KEY", raising=False)
    
    # Also patch the actual config variables as they have defaults
    with patch('src.utils.config.OPENAI_API_KEY', None), \
         patch('src.utils.config.CAL_API_KEY', None):
        # Validate should fail
        assert validate_config() is False

def test_validate_config_all_vars_present(monkeypatch):
    """Test that validate_config returns True if all required vars are present."""
    # Set dummy environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("CAL_API_KEY", "test_key")
    
    # Validate should succeed
    assert validate_config() is True 