#!/usr/bin/env python
"""
Debug script for testing natural language date/time extraction
"""

import re
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("debugtest")

def extract_natural_language_datetime(message: str):
    """
    Extract date and time from natural language expressions like "tomorrow at 2pm".
    """
    # Common patterns for dates and times
    tomorrow_pattern = r'\b(tomorrow|tmrw)\b'
    today_pattern = r'\b(today)\b'
    time_pattern = r'\b(\d{1,2})(:\d{2})?\s*(am|pm|AM|PM)\b'
    
    date_match = None
    time_match = None
    
    # Check for date expressions
    if re.search(tomorrow_pattern, message, re.IGNORECASE):
        # Calculate tomorrow's date
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = tomorrow.strftime("%Y-%m-%d")
        date_match = date_str
        logger.debug(f"Found 'tomorrow', set date to {date_match}")
    elif re.search(today_pattern, message, re.IGNORECASE):
        # Calculate today's date
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_match = date_str
        logger.debug(f"Found 'today', set date to {date_match}")
    
    # Check for time expressions
    time_search = re.search(time_pattern, message, re.IGNORECASE)
    if time_search:
        hour = int(time_search.group(1))
        minute = time_search.group(2)
        am_pm = time_search.group(3).lower()
        
        logger.debug(f"Found time components: hour={hour}, minute={minute}, am_pm={am_pm}")
        
        # Convert to 24-hour format
        if am_pm == 'pm' and hour < 12:
            hour += 12
        elif am_pm == 'am' and hour == 12:
            hour = 0
            
        logger.debug(f"After AM/PM conversion: hour={hour}")
            
        # Format the time (HH:MM)
        if minute:
            # Remove the colon if it's there
            minute = minute.replace(':', '')
            time_str = f"{hour:02d}:{minute}"
        else:
            time_str = f"{hour:02d}:00"
        
        time_match = time_str
        logger.debug(f"Final time string: {time_match}")
    
    if date_match and time_match:
        logger.info(f"Extracted date: {date_match}, time: {time_match} from message")
        return {
            "date": date_match,
            "time": time_match
        }
    
    return {}

def main():
    """Test function"""
    test_messages = [
        "I want to book a meeting tomorrow at 2pm",
        "Schedule something for today at 3:30pm",
        "Book a call tomorrow at 10am",
        "Let's meet today at 12pm for lunch",
        "No date or time here",
        "Invalid format: 25:00pm tomorrow"
    ]
    
    for message in test_messages:
        logger.info(f"Testing message: {message}")
        result = extract_natural_language_datetime(message)
        logger.info(f"Result: {result}")
        print("-" * 50)

if __name__ == "__main__":
    main() 