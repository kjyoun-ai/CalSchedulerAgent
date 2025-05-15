"""
Core chatbot implementation for the Cal.com Scheduler Agent.
This module will handle the OpenAI function calling and conversation management.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timedelta
import re
import traceback
import uuid

from src.utils.config import OPENAI_API_KEY, OPENAI_MODEL, logger
from src.bot.openai_integration import OpenAIFunctionCaller, INTENTS
from src.api.cal_api import CalAPIClient

class CalendarAgent:
    """
    Calendar Agent handles conversation with users and uses OpenAI's function
    calling to interact with the Cal.com API.
    """
    
    def __init__(self):
        """Initialize the Calendar Agent."""
        self.logger = logging.getLogger(__name__)
        if not OPENAI_API_KEY:
            self.logger.error("OpenAI API key is missing")
            raise ValueError("OpenAI API key is required")
        
        # Initialize the OpenAI function caller
        self.openai_caller = OpenAIFunctionCaller()
        
        # Initialize the Cal.com API client
        self.cal_api = None
        
        # Define the functions that will be exposed to the LLM
        self.available_functions = {
            "book_meeting": self.book_meeting,
            "list_events": self.list_events,
            "cancel_event": self.cancel_event,
            "reschedule_event": self.reschedule_event,
            "check_availability": self.get_available_slots
        }
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Conversation context variables
        self.conversation_context = {
            "current_user_email": None,
            "last_intent": None,
            "session_start_time": datetime.now().isoformat(),
            "last_action": None,
            "prev_action_success": None,
            "total_messages": 0,
            "total_actions": 0
        }
    
    async def initialize(self):
        """Initialize async components of the agent."""
        # Create the Cal.com API client using async context manager
        self.cal_api = CalAPIClient()
        
        # Test the API connection
        connection_success = await self.cal_api.test_api_connection()
        if not connection_success:
            self.logger.warning("Failed to connect to Cal.com API. Some features may not work correctly.")
    
    async def cleanup(self):
        """Clean up resources when the agent is no longer needed."""
        if self.cal_api and hasattr(self.cal_api, 'client') and self.cal_api.client:
            await self.cal_api.client.aclose()
    
    def _add_system_message_if_needed(self):
        """Add a system message to the conversation if one isn't present."""
        # Check if we already have a system message
        if any(msg.get("role") == "system" for msg in self.conversation_history):
            return
            
        # Add a comprehensive system message
        system_message = {
            "role": "system",
            "content": """You are a helpful calendar assistant that helps users manage their schedule through Cal.com.
            
You can help users with the following:
1. Booking new meetings (default is a 30-minute meeting with ID 2457598)
2. Checking availability for specific times and dates
3. Listing existing meetings
4. Canceling or rescheduling meetings

For meeting types:
- 30-minute meetings (ID: 2457598): These are the default meeting type
- Secret meetings (ID: 2457597): These are 15-minute meetings
- 15-minute meetings (ID: 2457599): Brief meetings for quick discussions

When handling time:
- ALWAYS verify that times are actually available before telling the user they are available
- NEVER claim that a time slot is available without checking the Cal.com API first
- If a user doesn't specify a time zone, assume they mean their local time
- Store times in ISO 8601 format with UTC timezone (e.g., 2025-05-14T20:30:00.000Z)
- For time ranges (like 3pm-5pm), find the earliest available slot in that range
- Parse time formats flexibly (3:00pm, 3pm, 15:00, etc.)
- When displaying times to users, ALWAYS convert from UTC to a readable format in PST

For booking requests:
- Always confirm the exact date, time, and duration when booking
- Check if the time is actually available before attempting to book
- If information is missing, ask for clarification
- After booking, provide confirmation with the booking details

IMPORTANT: Always explicitly verify times are available before mentioning them.
If a user asks for available slots in a time range, you MUST check if each time is available
before suggesting it.

When users ask for the earliest available time, provide the result immediately without 
asking for confirmation.

Maintain context throughout the conversation and refer back to previous messages when relevant.
            """
        }
        
        self.conversation_history.insert(0, system_message)
    
    def reset_conversation(self):
        """Reset the conversation history to a fresh state with only the system prompt."""
        self.conversation_history = []
        self._add_system_message_if_needed()
        self.conversation_context = {
            "total_messages": 0
        }
    
    def _update_context(self, intent: str, action_taken: str, success: bool):
        """Update the conversation context with new information."""
        self.conversation_context["last_intent"] = intent
        self.conversation_context["last_action"] = action_taken
        self.conversation_context["prev_action_success"] = success
        self.conversation_context["total_messages"] += 1
        
        if action_taken and action_taken != "none" and action_taken != "conversation_only":
            self.conversation_context["total_actions"] += 1
    
    async def process_message(self, message: str, user_email: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message and generate a response using OpenAI function calling.
        
        Args:
            message: The user's message
            user_email: Optional email of the user, to be stored in conversation context
            
        Returns:
            A dictionary containing the response and any action information
        """
        self.logger.info(f"Processing message: {message}")
        
        # Initialize the conversation context if it doesn't exist
        if not hasattr(self, "conversation_context"):
            self.conversation_context = {}
            
        # Initialize total_messages in conversation context if it doesn't exist
        if "total_messages" not in self.conversation_context:
            self.conversation_context["total_messages"] = 0
        
        # Store user email in conversation context if provided
        if user_email:
            self.conversation_context["current_user_email"] = user_email
            
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Check if message contains a request for earliest available time
        earliest_time_request = any(phrase in message.lower() for phrase in [
            "earliest", "soonest", "first available", "next available"
        ])
        
        # Parse date and duration from the message if it's a request for availability
        target_date = None
        duration = 30  # default duration in minutes
        
        # Check for date in the message using more robust patterns
        date_patterns = [
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            r'(\d{4})\s*(\d{1,2})[/-](\d{1,2})',    # YYYY MM/DD or YYYY MM-DD
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',   # MM-DD-YYYY or MM/DD/YYYY
            r'tomorrow',                           # tomorrow
            r'(\d{4})[\s,]+(\d{1,2})[\s/,]+(\d{1,2})', # 2025 5 15 or 2025, 5/15
            r'(\d{4})[\s,]+(\w+)[\s,]+(\d{1,2})',   # 2025 May 15 or 2025, May 15
            # For dates like "Monday, May 19, 2025"
            r'(?:\w+day,?\s+)?(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})'  # May 19, 2025 or Monday, May 19, 2025
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if pattern == r'tomorrow':
                    # Calculate tomorrow's date
                    tomorrow = datetime.now() + timedelta(days=1)
                    target_date = tomorrow.strftime("%Y-%m-%d")
                elif 'day' in pattern and 'May' not in pattern:  # New pattern for "Monday, May 19, 2025"
                    month_name = match.group(1)
                    day = int(match.group(2))
                    year = int(match.group(3))
                    
                    # Convert month name to number
                    month_names = ["january", "february", "march", "april", "may", "june", 
                                  "july", "august", "september", "october", "november", "december"]
                    try:
                        month_num = month_names.index(month_name.lower()) + 1
                        target_date = f"{year}-{month_num:02d}-{day:02d}"
                    except ValueError:
                        # Not a valid month name
                        continue
                elif 'month' in pattern or 'year' in pattern or (match.group(2).isalpha() if len(match.groups()) >= 3 else False):
                    # Handle patterns with month name
                    month_name = match.group(2)
                    month_names = ["january", "february", "march", "april", "may", "june", 
                                  "july", "august", "september", "october", "november", "december"]
                    if month_name.lower() in month_names:
                        month_num = month_names.index(month_name.lower()) + 1
                        year = int(match.group(1))
                        day = int(match.group(3))
                        target_date = f"{year}-{month_num:02d}-{day:02d}"
                else:
                    # Handle numeric date patterns
                    groups = match.groups()
                    if len(groups) == 3:
                        # Determine date format
                        if pattern.startswith(r'(\d{4})'):
                            # YYYY-MM-DD format
                            year = int(groups[0])
                            month = int(groups[1])
                            day = int(groups[2])
                        elif pattern.endswith(r'(\d{4})'):
                            # MM-DD-YYYY format
                            month = int(groups[0])
                            day = int(groups[1])
                            year = int(groups[2])
                        else:
                            continue  # Unknown format
                        
                        target_date = f"{year}-{month:02d}-{day:02d}"
                break
        
        # Default to tomorrow if no date found
        if not target_date:
            tomorrow = datetime.now() + timedelta(days=1)
            target_date = tomorrow.strftime("%Y-%m-%d")
        
        # Check for duration in the message
        duration_patterns = [
            r'(\d+)\s*(?:minute|min|minutes)',  # e.g., 30 minutes, 30min
            r'(\d+)\s*(?:hour|hr|hours|hrs)'    # e.g., 1 hour, 1hr
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                time_value = int(match.group(1))
                if 'hour' in pattern:
                    duration = time_value * 60  # convert hours to minutes
                else:
                    duration = time_value
                break
        
        # Adjust duration to match available event types (15, 30, or 60 minutes)
        if duration <= 15:
            duration = 15
        elif 15 < duration <= 30:
            duration = 30
        else:
            duration = 60
        
        # Check if this is a request for finding earliest availability
        if earliest_time_request and target_date:
            self.logger.info(f"Getting available slots for event type from {target_date} to with duration {duration} minutes")
            
            # Get available slots for the target date
            result = await self.get_available_slots(
                date_str=target_date,
                duration=duration
            )
            
            # Generate a response based on the result
            if result.get("status") == "success" and result.get("available_slots"):
                # Get the earliest available slot
                earliest_slot = result.get("available_slots")[0]
                earliest_display = earliest_slot.get("display")
                earliest_iso = earliest_slot.get("iso")
                
                response_text = f"The earliest available {duration}-minute slot on {target_date} is {earliest_display}. Would you like to book this time?"
            
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                return {
                    "response": response_text,
                    "action_taken": "get_available_slots",
                    "intent": "booking_intent",
                    "details": {
                        "status": "success",
                        "available_slots": result.get("available_slots"),
                        "earliest_slot": earliest_slot
                    },
                    "conversation_stats": {
                        "total_messages": self.conversation_context["total_messages"] + 1,
                        "total_actions": self.conversation_context["total_actions"] + 1
                    }
                }
            else:
                # Handle case where no slots are available
                error_msg = result.get("message", "No available slots found")
                
                if result.get("status") == "error":
                    response_text = "I'm sorry, but I'm currently experiencing connection issues with the calendar service. Please try again later or contact support if the problem persists."
                else:
                    response_text = f"I've checked for {duration}-minute slots on {target_date}, but there don't seem to be any available times. Would you like to try a different date or duration?"
                
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response_text,
                    "action_taken": "get_available_slots"
                })
                
                return {
                    "response": response_text,
                    "action_taken": "get_available_slots",
                    "intent": "booking_intent",
                    "details": {
                        "status": "error",
                        "message": error_msg
                    },
                    "conversation_stats": {
                        "total_messages": self.conversation_context["total_messages"] + 1,
                        "total_actions": self.conversation_context["total_actions"] + 1
                    }
                }
        
        # Use OpenAI function calling for regular conversation
        try:
            response = await self.openai_caller.process_with_function_calling(
                message=message,
                messages=self.conversation_history,
                functions={
                    "book_meeting": self.book_meeting,
                    "list_events": self.list_events,
                    "cancel_event": self.cancel_event,
                    "check_availability": self.get_available_slots
                }
            )
            
            # Add the response to the conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.get("response", "I'm sorry, I couldn't generate a response.")
            })
            
            # Update conversation context
            self.conversation_context["total_messages"] = self.conversation_context.get("total_messages", 0) + 1

            # Add action_taken if not present in the response
            if "action_taken" not in response:
                response["action_taken"] = "conversation_only"
                
            # Ensure details is present in the response
            if "details" not in response:
                response["details"] = response.get("function_result", {"status": "success", "action": "conversation"})
                
            # Ensure function_result is present in the response
            if "function_result" not in response:
                response["function_result"] = {"status": "success", "action": "conversation", "intent": response.get("intent", "unknown_intent")}
                
            # Ensure intent is present in the response
            if "intent" not in response:
                response["intent"] = "unknown_intent"
                
            # Return the response
            return response
        except Exception as e:
            self.logger.error(f"Error processing message with OpenAI function calling: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Provide a helpful error message to the user
            error_details = {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            
            error_response = {
                "response": "I apologize, but I'm having trouble processing your request. This might be due to a temporary issue with our calendar service. You can try again or be more specific with your request.",
                "action_taken": "error",
                "details": error_details,
                "function_result": error_details,
                "intent": "unknown_intent"
            }
            
            # Add the error response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": error_response["response"]
            })
            
            # Still increment the message count
            self.conversation_context["total_messages"] = self.conversation_context.get("total_messages", 0) + 1
            
            return error_response
    
    # Function implementations to be exposed to the LLM
    
    async def get_available_slots(self, date_str: str = None, duration: int = 30, span_days: int = 7, date: str = None, earliest: bool = False) -> Dict[str, Any]:
        """
        Get available time slots for a specific date and duration.
        
        Args:
            date_str: Date string in YYYY-MM-DD format (alias of date)
            date: Date string in YYYY-MM-DD format (alias of date_str) 
            duration: Meeting duration in minutes
            span_days: Number of days to check availability for
            earliest: If True, only return the earliest available slot
            
        Returns:
            Dictionary with available time slots
        """
        # Allow date parameter as alias for date_str
        if date and not date_str:
            date_str = date
            
        if not date_str:
            # Default to today if no date provided
            today = datetime.now()
            date_str = today.strftime("%Y-%m-%d")
            
        self.logger.info(f"Getting available slots for date {date_str} with duration {duration} minutes")
        self.logger.debug(f"DEBUG: get_available_slots called with date_str={date_str}, duration={duration}, span_days={span_days}, earliest={earliest}")
        
        try:
            # Initialize Cal.com API client
            cal_api = CalAPIClient()
            
            # First, find an appropriate event type based on duration
            self.logger.debug(f"DEBUG: Finding event type for duration {duration} minutes")
            event_type_result = await cal_api.get_event_type_by_duration(duration)
            self.logger.debug(f"DEBUG: Event type result: {json.dumps(event_type_result)}")
            
            if event_type_result.get("status") != "success":
                error_message = event_type_result.get("message", "Failed to find a suitable event type")
                self.logger.error(f"Error finding event type: {error_message}")
                return {
                    "status": "error",
                    "message": f"I couldn't find a suitable event type for a {duration}-minute meeting. {error_message}",
                    "technical_details": event_type_result,
                    "available_slots": []
                }
            
            event_type = event_type_result.get("event_type", {})
            event_type_id = str(event_type.get("id"))
            actual_duration = event_type.get("length", duration)
            
            self.logger.info(f"Using event type: ID {event_type_id}, title {event_type.get('title')}, duration {actual_duration} minutes")
            
            if actual_duration != duration:
                self.logger.info(f"Note: Using {actual_duration}-minute event type instead of requested {duration}-minute")
            
            # Get availability from Cal.com API
            self.logger.debug(f"DEBUG: Getting availability for event type {event_type_id}, date {date_str}")
            availability_response = await cal_api.get_availability(event_type_id, date_str)
            self.logger.debug(f"DEBUG: Availability response: {json.dumps(availability_response)}")
            
            # Check if there was an error getting availability
            if availability_response.get("status") == "error":
                self.logger.error(f"Error getting availability: {availability_response.get('message')}")
                self.logger.error(f"Details: {availability_response.get('details')}")
                return {
                    "status": "error",
                    "message": "Failed to retrieve availability from calendar",
                    "technical_details": availability_response.get("details", ""),
                    "available_slots": []
                }
            
            # Extract availability data from response
            availability_data = availability_response.get("availability", {})
            self.logger.debug(f"DEBUG: Availability data: {json.dumps(availability_data)}")
            
            # Process the date ranges from the API response
            date_ranges = availability_data.get("dateRanges", [])
            self.logger.info(f"Found {len(date_ranges)} date ranges")
            
            # Extract available time slots from date ranges
            available_slots = []
            max_slots_to_check = 10  # Limit the number of slots we check to avoid infinite loops
            slots_checked = 0
            
            if date_ranges:
                for date_range in date_ranges:
                    if slots_checked >= max_slots_to_check:
                        self.logger.info(f"Reached maximum slots to check ({max_slots_to_check})")
                        break
                        
                    start = date_range.get("start")
                    end = date_range.get("end")
                    
                    if not start or not end:
                        self.logger.warning(f"Invalid date range found: {date_range}")
                        continue
                    
                    try:
                        # Convert to datetime objects
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                        
                        # Only check the first few hours of availability to avoid excessive API calls
                        max_hours_to_check = 6
                        end_check_dt = min(end_dt, start_dt + timedelta(hours=max_hours_to_check))
                        
                        # Generate slots at intervals matching the requested duration
                        current_dt = start_dt
                        while current_dt + timedelta(minutes=actual_duration) <= end_check_dt and slots_checked < max_slots_to_check:
                            slots_checked += 1
                            
                            # Format the time for display
                            iso_time = current_dt.strftime('%Y-%m-%dT%H:%M:00.000Z')
                            display_time = self.format_datetime(iso_time)
                            
                            # Parse the formatted display time to extract date and time parts
                            display_parts = display_time.split(' at ')
                            formatted_date = display_parts[0] if len(display_parts) > 0 else ""
                            formatted_time = display_parts[1] if len(display_parts) > 1 else ""
                            
                            # Since we've simplified is_time_available to be permissive, we'll rely on display times
                            available_slots.append({
                                "iso": iso_time,
                                "display": display_time,
                                "formatted_date": formatted_date,
                                "formatted_time": formatted_time,
                                "event_type_id": event_type_id,
                                "duration": actual_duration
                            })
                            
                            # Move to the next slot
                            current_dt += timedelta(minutes=actual_duration)
                    except Exception as e:
                        self.logger.error(f"Error processing date range {date_range}: {str(e)}")
                        continue
            
            # Sort slots by time
            available_slots.sort(key=lambda x: x["iso"])
            self.logger.debug(f"DEBUG: Generated {len(available_slots)} available slots")
            for slot in available_slots[:5]:  # Log first 5 slots
                self.logger.debug(f"DEBUG: Slot: {slot['display']} (ISO: {slot['iso']})")
            
            # If earliest is True, only return the earliest slot
            if earliest and available_slots:
                earliest_slot = available_slots[0]
                self.logger.info(f"Returning earliest slot: {earliest_slot['display']} (ISO: {earliest_slot['iso']})")
                return {
                    "status": "success",
                    "message": "Found earliest available slot",
                    "available_slots": [earliest_slot],  # Return only the earliest slot
                    "event_type": {
                        "id": event_type_id,
                        "title": event_type.get("title"),
                        "duration": actual_duration
                    }
                }
            
            # Return the result with all slots (or limited to 10)
            return {
                "status": "success",
                "message": f"Found {len(available_slots)} available slots",
                "available_slots": available_slots[:10],  # Limit to first 10 slots
                "event_type": {
                    "id": event_type_id,
                    "title": event_type.get("title"),
                    "duration": actual_duration
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting available slots: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": "Failed to retrieve availability due to an error",
                "technical_details": str(e),
                "available_slots": []
            }
    
    async def book_meeting(self, date: str, time: str, event_type_id: str = None, name: str = "Anonymous User", reason: str = "Not specified", duration: int = 30, skip_availability_check: bool = True) -> Dict[str, Any]:
        """
        Book a meeting at the specified date and time with a reason.
        
        Args:
            date: The date for the meeting (YYYY-MM-DD)
            time: The time for the meeting (HH:MM)
            event_type_id: Optional ID of the event type to book (if not provided, will find by duration)
            name: The name of the attendee (defaults to "Anonymous User")
            reason: The reason for the meeting
            duration: The meeting duration in minutes (used to find appropriate event type if ID not provided)
            skip_availability_check: If True, skip checking availability and try booking directly
            
        Returns:
            A dictionary with booking details or error information
        """
        self.logger.info(f"Booking meeting on {date} at {time} for reason: {reason}")
        self.logger.debug(f"DEBUG: book_meeting called with date={date}, time={time}, event_type_id={event_type_id}, name={name}, reason={reason}, duration={duration}, skip_availability_check={skip_availability_check}")
        
        # Ensure Cal.com API client is initialized
        if not self.cal_api:
            await self.initialize()
        
        # Make sure we have an email to use
        email = self.conversation_context.get("current_user_email", "kjyoun3@gmail.com")
        if not email:
            return {
                "action": "book_meeting",
                "status": "error",
                "message": "No user email provided. Please provide an email to book a meeting."
            }
        
        try:
            # Clean up the time based on common formats and prepare it for API
            cleaned_time = time.strip()
            iso_datetime = None
            
            if time:  # Only process if time is provided
                # Handle timezone indicators
                timezone_offset = 0  # Default to UTC
                
                # Extract timezone if present and remove from time string
                if " PST" in cleaned_time or " PT" in cleaned_time:
                    # Pacific Time (UTC-8)
                    timezone_offset = 8  # Positive because we're adding hours to convert to UTC
                    cleaned_time = cleaned_time.replace(" PST", "").replace(" PT", "")
                    self.logger.info(f"Detected PST timezone, will add {timezone_offset} hours to convert to UTC")
                elif " EST" in cleaned_time or " ET" in cleaned_time:
                    # Eastern Time (UTC-5)
                    timezone_offset = 5  # Positive because we're adding hours to convert to UTC
                    cleaned_time = cleaned_time.replace(" EST", "").replace(" ET", "")
                    self.logger.info(f"Detected EST timezone, will add {timezone_offset} hours to convert to UTC")
                elif " GMT" in cleaned_time:
                    # GMT (UTC+0)
                    timezone_offset = 0
                    cleaned_time = cleaned_time.replace(" GMT", "")
                    self.logger.info("Detected GMT timezone, no conversion needed")
                elif " UTC" in cleaned_time:
                    # UTC (UTC+0)
                    timezone_offset = 0
                    cleaned_time = cleaned_time.replace(" UTC", "")
                    self.logger.info("Detected UTC timezone, no conversion needed")
                else:
                    # No timezone specified, default to PST (most likely user's local time)
                    timezone_offset = 8  # Positive because we're adding hours to convert to UTC
                    self.logger.info(f"No timezone specified, assuming PST and will add {timezone_offset} hours to convert to UTC")
                
                self.logger.info(f"After timezone processing: '{cleaned_time}' with offset {timezone_offset}")
                
                # Format date and time as ISO 8601 with UTC timezone
                try:
                    # Normalize time to 24-hour format if needed
                    is_pm = False
                    if "am" in cleaned_time.lower():
                        cleaned_time = cleaned_time.lower().replace("am", "").strip()
                    elif "pm" in cleaned_time.lower():
                        cleaned_time = cleaned_time.lower().replace("pm", "").strip()
                        is_pm = True
                    
                    # Handle missing colon
                    if ":" not in cleaned_time and cleaned_time.strip().isdigit():
                        # No colon and just a number, assume it's hour only
                        cleaned_time = f"{cleaned_time}:00"
                    
                    # Split the time into hours and minutes
                    if ":" in cleaned_time:
                        hours, minutes = cleaned_time.split(":")
                        hours = int(hours)
                        minutes = minutes.strip()
                        if minutes.isdigit():
                            minutes = int(minutes)
                        else:
                            minutes = 0
                    else:
                        # Best effort parsing for unusual formats
                        try:
                            hours = int(cleaned_time)
                            minutes = 0
                        except ValueError:
                            return {
                                "action": "book_meeting",
                                "status": "error",
                                "message": f"Could not parse time format: '{time}'. Please use a standard format like '3:00 PM'."
                            }
                    
                    # Adjust hours for PM
                    if is_pm and hours < 12:
                        hours += 12
                    elif not is_pm and hours == 12:
                        hours = 0
                    
                    # Convert from PST to UTC based on the timezone_offset we detected earlier
                    utc_hours = (hours + timezone_offset) % 24
                    self.logger.info(f"Converting local time {hours}:{minutes:02d} to UTC {utc_hours}:{minutes:02d} (adding {timezone_offset} hours)")
                    
                    # Format the date and time as ISO 8601
                    iso_datetime = f"{date}T{utc_hours:02d}:{minutes:02d}:00.000Z"
                    self.logger.info(f"Parsed local time '{time}' to UTC ISO format: {iso_datetime}")
                except Exception as e:
                    self.logger.error(f"Error formatting date and time: {str(e)}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    return {
                        "action": "book_meeting",
                        "status": "error",
                        "message": f"Invalid time format. Please provide time in a standard format like '3:00 PM'."
                    }
            else:
                # If no time is provided, we need to return an error
                return {
                    "action": "book_meeting",
                    "status": "error",
                    "message": "No time was specified for the booking. Please provide a time."
                }
            
            # First, find the appropriate event type if ID not provided
            if not event_type_id:
                self.logger.debug(f"DEBUG: Finding event type for duration {duration}")
                event_type_result = await self.cal_api.get_event_type_by_duration(duration)
                self.logger.debug(f"DEBUG: Event type result: {json.dumps(event_type_result)}")
                
                if event_type_result.get("status") != "success":
                    error_message = event_type_result.get("message", "Failed to find a suitable event type")
                    self.logger.error(f"Error finding event type: {error_message}")
                    return {
                        "action": "book_meeting",
                        "status": "error",
                        "message": f"I couldn't find a suitable event type for a {duration}-minute meeting. {error_message}",
                        "technical_details": event_type_result
                    }
                
                event_type = event_type_result.get("event_type", {})
                event_type_id = str(event_type.get("id"))
                actual_duration = event_type.get("length", duration)
                
                if actual_duration != duration:
                    self.logger.info(f"Using {actual_duration}-minute event type (ID: {event_type_id}) instead of requested {duration}-minute")
            
            self.logger.info(f"Using event_type_id: {event_type_id}")
            
            # Check if the time is available before booking
            self.logger.debug(f"DEBUG: Skip availability check: {skip_availability_check}")
            if not skip_availability_check:
                # Check if the time is available before booking
                is_available = await self.cal_api.is_time_available(event_type_id, iso_datetime)
                self.logger.debug(f"DEBUG: Time available check result: {is_available}")
                
                if not is_available:
                    # Get alternative slots to suggest
                    available_slots_result = await self.get_available_slots(date_str=date, duration=duration)
                    available_slots = []
                    
                    if available_slots_result.get("status") == "success":
                        available_slots = available_slots_result.get("available_slots", [])
                    
                    if available_slots:
                        # Provide alternative slots in the error message with proper PST formatting
                        alternatives = []
                        for slot in available_slots[:3]:
                            # The display time from get_available_slots should already be in PST format (e.g., "Tuesday, May 20, 2025 at 9:00 AM PST")
                            if "formatted_time" in slot:
                                alternatives.append(slot.get("formatted_time"))
                            elif "display" in slot:
                                display_time = slot.get("display", "")
                                if " at " in display_time:
                                    # Extract just the time part if it's in the format "Day, Month DD, YYYY at HH:MM AM/PM"
                                    time_part = display_time.split(" at ")[1]
                                    alternatives.append(time_part)
                                else:
                                    alternatives.append(display_time)
                        
                        alternatives_str = ", ".join(alternatives)
                        
                        # Get the date in a friendly format
                        try:
                            date_obj = datetime.strptime(date, "%Y-%m-%d")
                            friendly_date = date_obj.strftime("%A, %B %d, %Y")
                        except:
                            friendly_date = date
                            
                        friendly_message = f"That time slot ({time}) appears to be unavailable for {friendly_date}. This could be due to a host conflict or timezone mismatch. Alternative available times: {alternatives_str}"
                    else:
                        friendly_message = "That time slot is not available in the host's calendar. Please try another date or time."
                    return {
                        "action": "book_meeting",
                        "status": "error",
                        "message": friendly_message,
                        "technical_details": error_message,
                        "date": date,
                        "time": time
                    }
            else:
                # Even with skip_availability_check=True, we should verify time is within available ranges
                # to avoid the confusing "no_available_users_found_error"
                is_available = await self.cal_api.is_time_available(event_type_id, iso_datetime)
                
                # Add more logging for timezone debugging
                try:
                    dt = datetime.fromisoformat(iso_datetime.replace('Z', '+00:00'))
                    
                    # Convert from UTC to PST for display
                    pst_offset = timedelta(hours=-8)
                    dt_pst = dt + pst_offset
                    
                    pst_time = dt_pst.strftime("%I:%M %p PST")
                    self.logger.info(f"Checking availability for UTC time {iso_datetime} (converts to local time {pst_time})")
                except Exception as e:
                    self.logger.error(f"Error formatting time for display: {e}")
                    
                if not is_available:
                    self.logger.warning(f"Time {iso_datetime} is outside available ranges despite skip_availability_check=True")
                    return {
                        "action": "book_meeting",
                        "status": "error",
                        "message": f"That time slot ({time}) is outside the host's available hours. Please choose a time during business hours (typically 9AM-5PM PST)."
                    }
                self.logger.info("Verified time is within available ranges, proceeding with booking")
            
            # Call the Cal.com API to book the meeting
            self.logger.debug(f"DEBUG: Booking meeting with params: event_type_id={event_type_id}, start_time={iso_datetime}, name={name}, email={email}, reason={reason}")
            booking_result = await self.cal_api.book_event(
                event_type_id=event_type_id,
                start_time=iso_datetime,
                name=name,
                email=email,
                reason=reason
            )
            
            self.logger.info(f"Booking API result: {booking_result}")
            
            # Check if there was an error
            if "status" in booking_result and booking_result["status"] == "error":
                error_message = booking_result.get("message", "Unknown error")
                
                # Provide more user-friendly messages for common errors
                if "no_available_users_found_error" in error_message:
                    # This is most likely due to timezone or availability mismatch
                    # Get alternative slots to suggest
                    try:
                        available_slots_result = await self.get_available_slots(date_str=date, duration=duration)
                        available_slots = []
                        
                        if available_slots_result.get("status") == "success":
                            available_slots = available_slots_result.get("available_slots", [])
                        
                        if available_slots:
                            # Provide alternative slots in the error message with proper PST formatting
                            alternatives = []
                            for slot in available_slots[:3]:
                                # The display time from get_available_slots should already be in PST format (e.g., "Tuesday, May 20, 2025 at 9:00 AM PST")
                                if "formatted_time" in slot:
                                    alternatives.append(slot.get("formatted_time"))
                                elif "display" in slot:
                                    display_time = slot.get("display", "")
                                    if " at " in display_time:
                                        # Extract just the time part if it's in the format "Day, Month DD, YYYY at HH:MM AM/PM"
                                        time_part = display_time.split(" at ")[1]
                                        alternatives.append(time_part)
                                    else:
                                        alternatives.append(display_time)
                            
                            alternatives_str = ", ".join(alternatives)
                            
                            # Get the date in a friendly format
                            try:
                                date_obj = datetime.strptime(date, "%Y-%m-%d")
                                friendly_date = date_obj.strftime("%A, %B %d, %Y")
                            except:
                                friendly_date = date
                                
                            friendly_message = f"That time slot ({time}) appears to be unavailable for {friendly_date}. This could be due to a host conflict or timezone mismatch. Alternative available times: {alternatives_str}"
                        else:
                            friendly_message = "That time slot is not available in the host's calendar. Please try another date or time."
                    except Exception as e:
                        self.logger.error(f"Error getting alternative slots: {str(e)}")
                        friendly_message = "That time slot is not available in the host's calendar. Please try another time."
                elif "database" in error_message.lower():
                    friendly_message = "The calendar service is currently experiencing technical difficulties. Please try again later."
                elif "Invalid time slot" in error_message:
                    friendly_message = "That time slot is not valid or available. Please try a different time."
                elif "404" in error_message:
                    friendly_message = "The requested event type could not be found. Please check the event type ID."
                elif "401" in error_message or "403" in error_message:
                    friendly_message = "Authentication error when connecting to the calendar service. Please check your credentials."
                else:
                    friendly_message = f"Failed to book meeting: {error_message}"
                
                # Include the date in PST format for better user understanding
                try:
                    dt = datetime.fromisoformat(iso_datetime.replace('Z', '+00:00'))
                    pst_offset = timedelta(hours=-8)
                    dt_pst = dt + pst_offset
                    formatted_time_pst = dt_pst.strftime("%I:%M %p PST")
                    friendly_message += f" (Attempted to book at {formatted_time_pst} on {date})"
                except Exception as e:
                    self.logger.error(f"Error formatting time for error message: {str(e)}")
                
                return {
                    "action": "book_meeting",
                    "status": "error",
                    "message": friendly_message,
                    "technical_details": error_message,
                    "date": date,
                    "time": time
                }
            
            # For successful booking, extract the UID and format response
            elif "uid" in booking_result:
                # Format the response for the user
                start_time = booking_result.get("startTime", iso_datetime)
                formatted_time = self.format_datetime(start_time)
                
                return {
                    "action": "book_meeting",
                    "status": "success",
                    "message": f"Meeting booked successfully for {formatted_time}",
                    "date": date,
                    "time": time,
                    "reason": reason,
                    "booking_id": booking_result["uid"],
                    "formatted_time": formatted_time,
                    "details": booking_result
                }
            else:
                # Unexpected response format
                self.logger.error(f"Unexpected booking response: {booking_result}")
                return {
                    "action": "book_meeting",
                    "status": "error",
                    "message": "The booking system returned an unexpected response",
                    "technical_details": str(booking_result),
                    "date": date,
                    "time": time
                }
                
        except ValueError as e:
            # This would occur if the date/time format is invalid
            self.logger.error(f"Invalid date or time format: {str(e)}")
            return {
                "action": "book_meeting",
                "status": "error",
                "message": f"Invalid date or time format: {str(e)}"
            }
        except Exception as e:
            # Any other error
            self.logger.error(f"Error booking meeting: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "action": "book_meeting",
                "status": "error",
                "message": f"An unexpected error occurred while booking your meeting. Please try again later.",
                "technical_details": str(e),
                "date": date,
                "time": time
            }
    
    async def list_events(self, user_email: Optional[str] = None) -> Dict[str, Any]:
        """
        List all scheduled events for a user.
        
        Args:
            user_email: The email of the user to list events for
            
        Returns:
            A dictionary containing the list of events
        """
        # Use provided email or fall back to the one from conversation context
        email = user_email or self.conversation_context["current_user_email"]
        if not email:
            return {
                "action": "list_events",
                "status": "error",
                "message": "No user email provided. Please provide an email to list events."
            }
        
        self.logger.info(f"Listing events for user: {email}")
        
        # Ensure Cal.com API client is initialized
        if not self.cal_api:
            await self.initialize()
        
        try:
            # Call the Cal.com API to get the user's events
            events_data = await self.cal_api.list_bookings(email)
            
            # Check if there was an error
            if "status" in events_data and events_data["status"] == "error":
                return {
                    "action": "list_events",
                    "status": "error",
                    "message": f"Failed to list events: {events_data.get('message', 'Unknown error')}"
                }
            
            # Extract event information
            events = events_data.get("bookings", [])
            
            # Format the events for display
            formatted_events = []
            for event in events:
                formatted_events.append({
                    "id": event.get("uid", ""),
                    "title": event.get("title", "Untitled Event"),
                    "date": event.get("startTime", "").split("T")[0],
                    "time": event.get("startTime", "").split("T")[1].split(".")[0],
                    "duration": event.get("eventType", {}).get("length", 0),
                    "organizer": event.get("organizer", {}).get("name", "Unknown"),
                    "attendees": [attendee.get("name", "Unknown") for attendee in event.get("attendees", [])]
                })
            
            return {
                "action": "list_events",
                "status": "success",
                "message": f"Found {len(formatted_events)} events",
                "events": formatted_events
            }
        except Exception as e:
            self.logger.error(f"Error listing events: {e}")
            return {
                "action": "list_events",
                "status": "error",
                "message": f"Error listing events: {str(e)}"
            }
    
    async def cancel_event(self, event_id: str) -> Dict[str, Any]:
        """
        Cancel a specific event by ID.
        
        Args:
            event_id: The ID of the event to cancel
            
        Returns:
            A dictionary with the cancellation result
        """
        self.logger.info(f"Cancelling event: {event_id}")
        
        # Ensure Cal.com API client is initialized
        if not self.cal_api:
            await self.initialize()
        
        try:
            # Call the Cal.com API to cancel the event
            cancel_result = await self.cal_api.cancel_booking(event_id)
            
            # Check if there was an error
            if "status" in cancel_result and cancel_result["status"] == "error":
                return {
                    "action": "cancel_event",
                    "status": "error",
                    "message": f"Failed to cancel event: {cancel_result.get('message', 'Unknown error')}",
                    "event_id": event_id
                }
            
            return {
                "action": "cancel_event",
                "status": "success",
                "message": "Event cancelled successfully",
                "event_id": event_id
            }
        except Exception as e:
            self.logger.error(f"Error cancelling event: {e}")
            return {
                "action": "cancel_event",
                "status": "error",
                "message": f"Error cancelling event: {str(e)}",
                "event_id": event_id
            }
    
    async def reschedule_event(self, event_id: str, new_date: str, new_time: str) -> Dict[str, Any]:
        """
        Reschedule an event to a new date and time.
        
        Args:
            event_id: The ID of the event to reschedule
            new_date: The new date for the event (YYYY-MM-DD)
            new_time: The new time for the event (HH:MM)
            
        Returns:
            A dictionary with the rescheduling result
        """
        self.logger.info(f"Rescheduling event {event_id} to {new_date} at {new_time}")
        
        # Ensure Cal.com API client is initialized
        if not self.cal_api:
            await self.initialize()
        
        try:
            # Format the new date and time as ISO string
            new_start_time = f"{new_date}T{new_time}:00.000Z"
            
            # Call the Cal.com API to reschedule the event
            reschedule_result = await self.cal_api.reschedule_booking(event_id, new_start_time)
            
            # Check if there was an error
            if "status" in reschedule_result and reschedule_result["status"] == "error":
                return {
                    "action": "reschedule_event",
                    "status": "error",
                    "message": f"Failed to reschedule event: {reschedule_result.get('message', 'Unknown error')}",
                    "event_id": event_id
                }
            
            return {
                "action": "reschedule_event",
                "status": "success",
                "message": "Event rescheduled successfully",
                "event_id": event_id,
                "new_date": new_date,
                "new_time": new_time
            }
        except Exception as e:
            self.logger.error(f"Error rescheduling event: {e}")
            return {
                "action": "reschedule_event",
                "status": "error",
                "message": f"Error rescheduling event: {str(e)}",
                "event_id": event_id
            }
            
    async def test_chat(self, user_email: str = "test@example.com", verbose: bool = True) -> Dict[str, Any]:
        """
        Test the chatbot with a sample conversation flow.
        
        Args:
            user_email: Email to use for the test
            verbose: Whether to print conversation details
            
        Returns:
            Dictionary with test results
        """
        test_messages = [
            "Hello, I need to manage my calendar",
            "I'd like to book a meeting for next week",
            "Can you list my current bookings?",
            "I need to cancel my meeting on Thursday"
        ]
        
        test_results = {
            "messages": [],
            "intents_recognized": [],
            "actions_taken": [],
            "successful_actions": 0,
            "failed_actions": 0
        }
        
        # Reset conversation state
        self.conversation_history = []
        self.conversation_context = {
            "current_user_email": user_email,
            "last_intent": None,
            "session_start_time": datetime.now().isoformat(),
            "last_action": None,
            "prev_action_success": None,
            "total_messages": 0,
            "total_actions": 0
        }
        
        if verbose:
            print(f"\n===== Starting Chatbot Test with {user_email} =====\n")
        
        for i, message in enumerate(test_messages):
            if verbose:
                print(f"\nUSER: {message}")
                
            # Process the message
            response = await self.process_message(message)
            
            if verbose:
                print(f"BOT: {response}")
                
            # Record test data
            test_results["messages"].append({
                "user": message,
                "assistant": response
            })
            test_results["intents_recognized"].append(response.get("intent", "unknown"))
            test_results["actions_taken"].append(response.get("action_taken", "none"))
            
            # Count successful and failed actions
            if response.get("action_taken", "none") not in ["none", "conversation_only", "error"]:
                if response.get("details", {}).get("status", "") == "success":
                    test_results["successful_actions"] += 1
                else:
                    test_results["failed_actions"] += 1
                    
        # Add conversation stats
        test_results["conversation_stats"] = {
            "total_messages": self.conversation_context["total_messages"],
            "total_actions": self.conversation_context["total_actions"]
        }
        
        if verbose:
            print("\n===== Test Complete =====")
            print(f"Total messages: {test_results['conversation_stats']['total_messages']}")
            print(f"Intents recognized: {test_results['intents_recognized']}")
            print(f"Actions taken: {test_results['actions_taken']}")
            print(f"Successful actions: {test_results['successful_actions']}")
            print(f"Failed actions: {test_results['failed_actions']}")
        
        return test_results

    def format_datetime(self, iso_datetime: str) -> str:
        """
        Format ISO 8601 datetime string into a human-readable format.
        Example: 2025-05-15T15:00:00.000Z -> Thursday, May 15, 2025 at 8:00 AM PST
        
        Args:
            iso_datetime: ISO 8601 formatted datetime string
            
        Returns:
            Human-readable formatted datetime string
        """
        try:
            # Parse the ISO datetime string
            dt = datetime.fromisoformat(iso_datetime.replace('Z', '+00:00'))
            
            # Convert to PST (UTC-8)
            pst_offset = timedelta(hours=-8)
            dt_pst = dt + pst_offset
            
            # Format the datetime in a user-friendly way
            weekday = dt_pst.strftime('%A')
            month = dt_pst.strftime('%B')
            day = dt_pst.day
            year = dt_pst.year
            
            # Convert to 12-hour format
            hour = dt_pst.hour % 12
            if hour == 0:
                hour = 12
            am_pm = "AM" if dt_pst.hour < 12 else "PM"
            
            # Format the final string
            return f"{weekday}, {month} {day}, {year} at {hour}:{dt_pst.minute:02d} {am_pm} PST"
            
        except Exception as e:
            self.logger.error(f"Error formatting datetime: {str(e)}")
            # Return the original string if there's an error
            return iso_datetime

    async def book_appointment(
            self, 
            event_type_id: str,
            start_time: str,
            name: str = None,
            email: str = None,
            notes: str = None
        ) -> Dict[str, Any]:
            """
            Book an appointment using the Cal.com API.
            
            Args:
                event_type_id: The ID of the event type to book
                start_time: ISO 8601 formatted start time
                name: The name of the attendee (optional)
                email: The email of the attendee (optional)
                notes: Notes for the booking (optional)
                
            Returns:
                A dictionary with the result of the booking attempt
            """
            self.logger.info(f"Attempting to book meeting at {start_time}")
            
            # Default values for testing if not provided
            if not name:
                name = "Test User"
            if not email:
                email = "test@example.com"
            if not notes:
                notes = "Booked via chatbot"
            
            # Check if the time is available before booking
            is_available = await self.cal_api.is_time_available(event_type_id, start_time)
            
            if not is_available:
                formatted_time = self.format_datetime(start_time)
                return {
                    "response": f"I'm sorry, but the time slot at {formatted_time} is not available. Would you like to check for another time?",
                    "action_taken": "book_appointment",
                    "intent": "booking_intent",
                    "details": {
                        "status": "error",
                        "message": "Time slot not available",
                        "requested_time": start_time
                    }
                }
            
            try:
                # Call the API to book the appointment
                result = await self.cal_api.book_slot(
                    event_type_id=event_type_id,
                    start_time=start_time,
                    name=name,
                    email=email,
                    notes=notes
                )
                
                if result.get("status") == "success":
                    booking_info = result.get("booking", {})
                    booking_time = booking_info.get("startTime", start_time)
                    formatted_time = self.format_datetime(booking_time)
                    
                    response_text = f"Great! I've successfully booked your {booking_info.get('duration', '30')} minute meeting for {formatted_time}. You should receive a confirmation email shortly. The meeting ID is {booking_info.get('id')}."
                    
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    return {
                        "response": response_text,
                        "action_taken": "book_appointment",
                        "intent": "booking_intent",
                        "details": {
                            "status": "success",
                            "booking_info": booking_info,
                            "formatted_time": formatted_time
                        },
                        "conversation_stats": {
                            "total_messages": self.conversation_context["total_messages"] + 1,
                            "total_actions": self.conversation_context["total_actions"] + 1
                        }
                    }
                else:
                    error_message = result.get("message", "Unknown error occurred during booking")
                    if "403" in error_message or "not authorized" in error_message.lower():
                        response_text = f"I'm sorry, but I'm experiencing connection issues with the booking service. Please try again later or contact support."
                    else:
                        formatted_time = self.format_datetime(start_time)
                        response_text = f"I'm sorry, but I couldn't book the appointment for {formatted_time}. {error_message}"
                    
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    return {
                        "response": response_text,
                        "action_taken": "book_appointment",
                        "intent": "booking_intent",
                        "details": {
                            "status": "error",
                            "message": error_message,
                            "requested_time": start_time
                        },
                        "conversation_stats": {
                            "total_messages": self.conversation_context["total_messages"] + 1,
                            "total_actions": self.conversation_context["total_actions"] + 1
                        }
                    }
            except Exception as e:
                self.logger.error(f"Error booking appointment: {str(e)}")
                
                response_text = f"I'm sorry, but I encountered an error while trying to book your appointment. Please try again later or contact support."
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                return {
                    "response": response_text,
                    "action_taken": "book_appointment",
                    "intent": "booking_intent",
                    "details": {
                        "status": "error",
                        "message": str(e),
                        "requested_time": start_time
                    },
                    "conversation_stats": {
                        "total_messages": self.conversation_context["total_messages"] + 1,
                        "total_actions": self.conversation_context["total_actions"] + 1
                    }
                } 