"""
OpenAI Function Calling Integration Module
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import asyncio
import traceback
import pytz  # For timezone handling
from datetime import datetime, timedelta

from langchain import LLMChain
from langchain.llms import BaseLLM
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.utils.config import OPENAI_API_KEY, OPENAI_MODEL

# Define common intents for better categorization
INTENTS = {
    "BOOKING": "booking_intent",
    "CANCELLATION": "cancellation_intent",
    "RESCHEDULING": "rescheduling_intent",
    "LISTING": "listing_intent",
    "AVAILABILITY": "availability_intent",
    "GREETING": "greeting_intent",
    "HELP": "help_intent",
    "CONFIRMATION": "confirmation_intent",
    "UNKNOWN": "unknown_intent"
}

# Monkey patch LLMChain to properly handle coroutines
original_call = LLMChain._call
async def patched_call(self, inputs, run_manager=None):
    try:
        result = await original_call(self, inputs, run_manager)
        
        # Check if any of the results are coroutines and await them
        for key, value in result.items():
            if asyncio.iscoroutine(value):
                result[key] = await value
                
        return result
    except Exception as e:
        logging.error(f"Error in patched LLMChain._call: {e}")
        return {"output": f"Error: {str(e)}"}

# Apply the monkey patch
LLMChain._call = patched_call

class OpenAIFunctionCaller:
    """
    Handler class for OpenAI function calling using LangChain.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the OpenAI function caller.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from environment)
            model: OpenAI model name (defaults to OPENAI_MODEL from environment)
        """
        self._logger = logging.getLogger(__name__)
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize the OpenAI API client
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            temperature=0.7
        )
        
        self._logger.info(f"Initialized LangChain ChatOpenAI client with model {self.model}")
        
        # Define the system message for intent classification
        self.intent_system_message = """
        You are an intent classifier for a calendar assistant. Your task is to identify the primary intent of the user's message.
        
        Possible intents are:
        - booking_intent: User wants to book a new meeting or appointment
        - cancellation_intent: User wants to cancel an existing meeting
        - listing_intent: User wants to list or view their schedule/events
        - rescheduling_intent: User wants to reschedule an existing meeting
        - availability_intent: User wants to check available slots or times
        - greeting_intent: User is just saying hello or starting a conversation
        - unknown_intent: Intent cannot be determined
        
        Respond with ONLY the intent name, no other text.
        """

    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger

    def _make_tools(self, functions: Dict[str, Callable]) -> list:
        """
        Create LangChain StructuredTool objects from Python functions.
        """
        tools = []
        for name, func in functions.items():
            tools.append(StructuredTool.from_function(
                func=func,
                name=name,
                description=func.__doc__ or f"Tool for {name}"
            ))
        return tools
    
    async def _identify_intent_from_history(self, messages: List[Dict[str, str]]) -> str:
        """
        Identify the intent using the complete conversation history with emphasis on recent messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            The identified intent
        """
        try:
            # If we have no messages or just one, use basic intent detection with just that message
            if not messages or len(messages) <= 1:
                last_message = messages[-1]["content"] if messages else ""
                
                # Create a single message for the LLM
                lc_messages = [HumanMessage(content=last_message)]
                system_message = SystemMessage(content=self.intent_system_message)
                intent_messages = [system_message] + lc_messages
                
                # Create a separate LLM instance with lower temperature for classification
                intent_llm = ChatOpenAI(
                    model=self.model,
                    openai_api_key=self.api_key,
                    temperature=0.2  # Lower temperature for more consistent answers
                )
                
                # Call the model
                self._logger.info("Calling LLM for intent classification on single message")
                intent_response = await intent_llm.agenerate([intent_messages])
                
                # Extract the intent from the response
                intent_text = intent_response.generations[0][0].text.strip().lower()
                self._logger.info(f"LLM intent classification result for single message: {intent_text}")
                
                # Return the intent
                return self._validate_intent(intent_text)
                
            # For multiple messages, use the most recent 5 messages for context
            # Convert messages to LangChain format
            lc_messages = self._convert_to_langchain_messages(messages[-5:])  # Focus on the most recent 5 messages
            
            # Create an intention classification prompt
            system_message = SystemMessage(content=self.intent_system_message)
            
            # Add the system message to the beginning
            intent_messages = [system_message] + lc_messages
            
            # Create a separate LLM instance with lower temperature for classification
            intent_llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key, 
                temperature=0.2  # Lower temperature for more consistent answers
            )
            
            # Call the model
            self._logger.info("Calling LLM for intent classification on conversation history")
            intent_response = await intent_llm.agenerate([intent_messages])
            
            # Extract the intent from the response
            intent_text = intent_response.generations[0][0].text.strip().lower()
            self._logger.info(f"LLM intent classification result: {intent_text}")
            
            return self._validate_intent(intent_text)
            
        except Exception as e:
            self._logger.error(f"Error in LLM intent classification: {str(e)}")
            self._logger.error(traceback.format_exc())
            return INTENTS["UNKNOWN"]

    def _validate_intent(self, intent_text: str) -> str:
        """
        Validate and normalize the intent text from the LLM.
        
        Args:
            intent_text: The raw intent text from the LLM
            
        Returns:
            A validated intent string
        """
        # Valid intents list
        valid_intents = [
            "booking_intent", "cancellation_intent", "listing_intent", 
            "rescheduling_intent", "availability_intent", "greeting_intent",
            "unknown_intent", "date_time_intent", "confirmation_intent"
        ]
        
        # Extract just the intent name if there's additional text
        # Try to match one of the valid intents
        matched_intent = None
        for intent in valid_intents:
            if intent in intent_text:
                matched_intent = intent
                break
        
        if matched_intent:
            return matched_intent
        else:
            # If no valid intent found, return unknown
            self._logger.warning(f"Couldn't match '{intent_text}' to a valid intent, returning unknown_intent")
            return "unknown_intent"
    
    async def process_with_function_calling(self, message: str, messages: List[Dict[str, str]], functions: Dict[str, Callable] = None) -> Dict[str, Any]:
        """
        Process a message with function calling capabilities.
        
        Args:
            message: The current user message
            messages: A list of all messages in the conversation 
            functions: A dictionary of functions that can be called
            
        Returns:
            A dictionary containing response text and function result information
        """
        if not functions:
            functions = {}
            
        # First, analyze the intent using the entire conversation history
        if not message:
            return {"response": "Please provide a message.", "function_result": {"status": "error", "message": "No user message found"}}
            
        # Identify the intent using our conversation-based intent recognizer
        intent = await self._identify_intent_from_history(messages)
        self._logger.info(f"Identified intent from conversation history: {intent}")
        
        # For simple time responses, check if they're part of an ongoing booking conversation
        if self._is_simple_time_response(message) and len(messages) > 2:
            # Check the previous user messages (up to 3) for booking intent
            prev_intents = []
            for i, msg in enumerate(reversed(messages[:-1])):
                if i >= 3:  # Only check last 3 messages
                    break
                if msg["role"] == "user":
                    prev_intent = await self._identify_intent_from_history([msg])
                    prev_intents.append(prev_intent)
            
            # If any previous message had booking or availability intent, this is likely a booking intent
            if any(prev_intent in ["booking_intent", "availability_intent"] for prev_intent in prev_intents):
                intent = "booking_intent"
                self._logger.info(f"Detected simple time response in a booking context, updated intent: {intent}")
        
        # If we detect a booking intent, use the function calling capability
        if intent == "booking_intent" and "book_meeting" in functions:
            self._logger.info("Processing booking intent with function calling")
            book_meeting_func = functions["book_meeting"]
            
            # First, try to extract date/time using LLM for the current message
            dt_params = await self._extract_datetime_with_llm(message)
            booking_params = dt_params.copy()  # Start with LLM-extracted date/time
            
            # If LLM couldn't extract date/time, try ISO format as fallback
            if not booking_params or not ("date" in booking_params and "time" in booking_params):
                iso_params = self._extract_iso_datetime(message)
                if iso_params:
                    booking_params.update(iso_params)
            
            # Special handling for follow-up messages that might just specify a time
            # but depend on context from previous messages
            if self._is_simple_time_response(message) and "date" not in booking_params:
                self._logger.info("Detected simple time response, checking conversation history")
                
                # First, try to extract date from history using LLM
                for i, msg in enumerate(reversed(messages[:-1])):
                    if i >= 5:  # Only check last 5 messages
                        break
                    if msg["role"] == "user":
                        hist_dt_params = await self._extract_datetime_with_llm(msg["content"])
                        if hist_dt_params and "date" in hist_dt_params:
                            booking_params["date"] = hist_dt_params["date"]
                            booking_params["time"] = message.strip()  # Use current message as time
                            self._logger.info(f"Using date {hist_dt_params['date']} from conversation history with time {message}")
                            break
            
            # If we have date and time from either extraction method, process directly
            if "date" in booking_params and "time" in booking_params:
                time_str = booking_params["time"]
                self._logger.debug(f"DEBUG: Processing direct booking with date={booking_params['date']} and time={time_str}")
                
                try:
                    # Process time format if needed
                    # Check if time_str contains "at" or timezone indicators
                    if ' at ' in time_str or any(f" {tz}" in time_str.lower() for tz in ["pst", "est", "cst", "mst", "utc", "gmt"]):
                        time_str = self._normalize_time_format(time_str)
                        self._logger.info(f"Normalized time string from '{booking_params['time']}' to '{time_str}'")
                    else:
                        time_str = self._normalize_time_format(time_str)
                    
                    booking_params["time"] = time_str
                    
                    self._logger.info(f"Processing booking with params: {booking_params}")
                    
                    try:
                        # Call the booking function with extracted parameters
                        self._logger.info(f"info: Calling book_meeting_func with parameters: {booking_params}")
                        booking_result = await book_meeting_func(
                            date=booking_params.get("date"),
                            time=booking_params.get("time"),
                            name=booking_params.get("name", "Anonymous User"),
                            reason=booking_params.get("reason", "Not specified"),
                            duration=booking_params.get("duration", 30),
                            skip_availability_check=True
                        )
                        self._logger.info(f"Booking result: {booking_result}")
                        
                        # Check if booking was successful
                        if booking_result.get("status") == "success":
                            response_text = f"Great! I've booked your meeting for {booking_result.get('formatted_time', booking_params.get('date') + ' at ' + booking_params.get('time'))}"
                            if booking_params.get("reason"):
                                response_text += f" regarding \"{booking_params.get('reason')}\""
                            response_text += ". Is there anything else you need help with?"
                            
                            # Return both the response text and the function result
                            return {
                                "response": response_text, 
                                "function_result": booking_result
                            }
                        else:
                            # Booking failed, return the error message
                            error_message = booking_result.get("message", "An unknown error occurred")
                            response_text = f"I couldn't book your meeting: {error_message} Would you like to try a different time?"
                            return {
                                "response": response_text, 
                                "function_result": booking_result
                            }
                    except Exception as e:
                        self._logger.error(f"Error calling booking function: {str(e)}")
                        self._logger.error(traceback.format_exc())
                        return {
                            "response": "Sorry, I encountered an error while trying to book your meeting. Please try again.",
                            "function_result": {"status": "error", "message": str(e)}
                        }
                    
                except Exception as e:
                    self._logger.error(f"Error processing booking request: {str(e)}")
                    self._logger.error(traceback.format_exc())
                    return {
                        "response": "I couldn't understand the time format. Please specify the time in a format like '3:00 PM' or '15:00'.",
                        "function_result": {"status": "error", "message": str(e)}
                    }
            
            # If we don't have enough information for booking, use function calling to get it
            self._logger.info("Not enough booking information, using function calling to extract parameters")
            
            try:
                # Use ChatOpenAI and LangChain tools to extract booking parameters
                # Convert to LangChain messages
                self._logger.debug(f"DEBUG: Converting {len(messages)} messages to LangChain format for function calling")
                lc_messages = self._convert_to_langchain_messages(messages)
                if not lc_messages:
                    self._logger.warning("Failed to convert messages to LangChain format")
                    lc_messages = [HumanMessage(content=message)]
                
                self._logger.debug(f"DEBUG: Converted {len(lc_messages)} messages to LangChain format")
                
                # Create a new OpenAI instance with function calling
                function_llm = ChatOpenAI(
                    model=self.model,
                    temperature=0.2,
                    openai_api_key=self.api_key
                )
                
                # Define the booking function schema for the LLM
                date_today = datetime.today().strftime("%Y-%m-%d")
                function_definitions = [
                    {
                        "name": "book_meeting",
                        "description": "Book a meeting or appointment on the calendar",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "date": {
                                    "type": "string",
                                    "description": f"The date for the meeting in YYYY-MM-DD format. Today is {date_today}."
                                },
                                "time": {
                                    "type": "string",
                                    "description": "The time for the meeting in 24-hour format (HH:MM) or 12-hour format (e.g., '3:00 PM')"
                                },
                                "duration": {
                                    "type": "number",
                                    "description": "Duration of the meeting in minutes (default is 30 minutes)",
                                    "default": 30
                                },
                                "name": {
                                    "type": "string",
                                    "description": "The name of the person booking the meeting"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "The reason or topic for the meeting"
                                }
                            },
                            "required": ["date", "time"]
                        }
                    }
                ]
                
                self._logger.debug(f"DEBUG: Using function definitions: {json.dumps(function_definitions)}")
                
                # Prepare the model with functions
                function_llm = function_llm.bind(
                    functions=function_definitions
                )
                
                # Get a response with function calling
                self._logger.info("Calling function LLM to extract booking parameters")
                function_response = await function_llm.ainvoke(lc_messages)
                self._logger.debug(f"DEBUG: Function response: {function_response}")
                
                # Check if the model called a function
                if hasattr(function_response, 'additional_kwargs') and 'function_call' in function_response.additional_kwargs:
                    function_call = function_response.additional_kwargs['function_call']
                    self._logger.info(f"Function call detected: {function_call['name']}")
                    
                    if function_call['name'] == 'book_meeting':
                        # Parse the arguments
                        args = json.loads(function_call['arguments'])
                        self._logger.debug(f"DEBUG: Book meeting arguments: {args}")
                        
                        # Format time if needed
                        if 'time' in args:
                            args['time'] = self._normalize_time_format(args['time'])
                        
                        # Now make the actual booking
                        try:
                            self._logger.info(f"Making booking with args: {args}")
                            
                            # Clean the time string if it contains "at" or timezone indicators
                            if 'time' in args:
                                time_str = args.get('time')
                                # Handle complex time formats with 'at' or timezone indicators
                                if ' at ' in time_str or any(f" {tz}" in time_str.lower() for tz in ["pst", "est", "cst", "mst", "utc", "gmt"]):
                                    args['time'] = self._normalize_time_format(time_str)
                                    self._logger.info(f"Normalized time string from '{time_str}' to '{args['time']}'")
                            
                            booking_result = await book_meeting_func(
                                date=args.get('date'),
                                time=args.get('time'),
                                name=args.get('name', 'Anonymous User'),
                                reason=args.get('reason', 'Not specified'),
                                duration=args.get('duration', 30),
                                skip_availability_check=True
                            )
                            self._logger.debug(f"DEBUG: Booking result: {booking_result}")
                            
                            # Check if booking was successful
                            if booking_result.get("status") == "success":
                                response_text = f"Great! I've booked your meeting for {booking_result.get('formatted_time', args.get('date') + ' at ' + args.get('time'))}"
                                if args.get("reason"):
                                    response_text += f" regarding \"{args.get('reason')}\""
                                response_text += ". Is there anything else you need help with?"
                                
                                return {
                                    "response": response_text, 
                                    "function_result": booking_result
                                }
                            else:
                                # Booking failed, return the error message
                                error_message = booking_result.get("message", "An unknown error occurred")
                                response_text = f"I couldn't book your meeting: {error_message} Would you like to try a different time?"
                                return {
                                    "response": response_text, 
                                    "function_result": booking_result
                                }
                        except Exception as e:
                            self._logger.error(f"Error making booking: {str(e)}")
                            self._logger.error(traceback.format_exc())
                            return {
                                "response": "Sorry, I encountered an error while trying to book your meeting. Please try again.",
                                "function_result": {"status": "error", "message": str(e)}
                            }
                
                # If no function call or if it failed, generate a text response
                response_text = self._extract_response_content(function_response)
                self._logger.info(f"Generated response text (no function call): {response_text}")
                return {"response": response_text}
                
            except Exception as e:
                self._logger.error(f"Error in booking intent processing: {str(e)}")
                self._logger.error(traceback.format_exc())
                return {"response": f"I'm sorry, I encountered an error while processing your booking request: {str(e)}"}
        
        # For cancellation intents, extract the event information
        elif intent == "cancellation_intent":
            self._logger.info("Processing cancellation intent")
            
            # First check if there's an event ID mentioned
            event_id_match = re.search(r'event\s+id\s*(?:[:=]?)?\s*(\w+)', message, re.IGNORECASE)
            cancellation_params = {}
            
            if event_id_match:
                # Direct cancellation with Event ID
                event_id = event_id_match.group(1)
                cancellation_params["event_id"] = event_id
                self._logger.info(f"Found event ID for cancellation: {event_id}")
            else:
                # Try to extract date/time for event lookup
                # First try ISO format
                datetime_params = self._extract_iso_datetime(message)
                
                # If that fails, try natural language
                if not datetime_params:
                    datetime_params = self._extract_natural_language_datetime(message)
                
                if datetime_params and "date" in datetime_params:
                    cancellation_params["date"] = datetime_params.get("date")
                    if "time" in datetime_params:
                        cancellation_params["time"] = datetime_params.get("time")
            
            # If we have parameters, try to cancel or look up the event
            cancel_event_func = functions.get("cancel_event")
            list_events_func = functions.get("list_events")
            
            if cancellation_params and "event_id" in cancellation_params and cancel_event_func:
                # We have an event ID, proceed with cancellation
                try:
                    event_id = cancellation_params["event_id"]
                    
                    # Check if this is a confirmation message
                    is_confirmation = re.search(r'\b(yes|confirm|proceed|go ahead|do it)\b', message, re.IGNORECASE)
                    
                    if not is_confirmation:
                        # Ask for confirmation first
                        return {"response": f"Are you sure you want to cancel the event with ID {event_id}? Please confirm.", "function_result": {"action": "cancel_confirmation_needed", "status": "pending", "event_id": event_id}}
                    
                    # User confirmed, proceed with cancellation
                    cancel_result = await cancel_event_func(event_id=event_id)
                    
                    self._logger.info(f"Cancellation result: {cancel_result}")
                    
                    if cancel_result.get("status") == "success":
                        response_text = f"I've successfully cancelled your event with ID {event_id}."
                    else:
                        error_msg = cancel_result.get("message", "Unknown error")
                        if "500" in error_msg and "database" in error_msg.lower():
                            response_text = "I'm sorry, but the calendar service is currently experiencing technical difficulties. Please try again later."
                        elif "not found" in error_msg.lower():
                            response_text = f"I couldn't find an event with ID {event_id}. Please check the ID and try again."
                        else:
                            response_text = f"I couldn't cancel the event: {error_msg}."
                    
                    return {"response": response_text, "function_result": cancel_result}
                except Exception as e:
                    self._logger.error(f"Error cancelling event: {e}", exc_info=True)
                    return {"response": f"I encountered an error while trying to cancel the event: {str(e)}.", "function_result": {"status": "error", "message": str(e), "action": "cancel_event"}}
            
            elif cancellation_params and "date" in cancellation_params and list_events_func:
                # We have a date but no ID, try to list events for that date to help user select the right one
                try:
                    # Get the user email from a prior message if available
                    user_email = None
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    for msg in reversed(messages):
                        if msg["role"] == "user":
                            email_match = re.search(email_pattern, msg["content"])
                            if email_match:
                                user_email = email_match.group(0)
                                break
                    
                    if not user_email:
                        return {"response": "I need your email address to look up your events. Please provide your email.", "function_result": {"action": "email_needed", "status": "pending"}}
                    
                    # List events for the user
                    events_result = await list_events_func(user_email=user_email)
                    
                    if events_result.get("status") == "success" and events_result.get("events"):
                        events = events_result.get("events", [])
                        date = cancellation_params.get("date")
                        
                        # Filter events by date
                        matching_events = [e for e in events if e.get("date") == date]
                        
                        # If time is provided, refine further
                        if "time" in cancellation_params and matching_events:
                            time = cancellation_params.get("time")
                            # Find events within 30 minutes of the specified time
                            close_events = []
                            for event in matching_events:
                                event_time = event.get("time", "")
                                if event_time.startswith(time[:2]):  # Match by hour
                                    close_events.append(event)
                            
                            if close_events:
                                matching_events = close_events
                        
                        if matching_events:
                            # Format events for display
                            events_text = "\n".join([
                                f"- {e.get('title')} at {e.get('time')} (ID: {e.get('id')})"
                                for e in matching_events
                            ])
                            
                            if len(matching_events) == 1:
                                event_id = matching_events[0].get("id")
                                return {"response": f"I found this event for {date}:\n{events_text}\n\nWould you like me to cancel this event? Please confirm.", "function_result": {"action": "cancel_confirmation_needed", "status": "pending", "event_id": event_id}}
                            else:
                                return {"response": f"I found these events for {date}:\n{events_text}\n\nPlease specify which event you'd like to cancel by providing the event ID.", "function_result": {"action": "event_selection_needed", "status": "pending", "events": matching_events}}
                        else:
                            return {"response": f"I couldn't find any events scheduled for {date}. Would you like to list all your upcoming events?", "function_result": {"action": "no_events_found", "status": "pending"}}
                    else:
                        error_msg = events_result.get("message", "Unknown error")
                        return {"response": f"I couldn't retrieve your events: {error_msg}. Please try again later.", "function_result": events_result}
                except Exception as e:
                    self._logger.error(f"Error looking up events for cancellation: {e}", exc_info=True)
                    return {"response": f"I encountered an error while trying to look up your events: {str(e)}.", "function_result": {"status": "error", "message": str(e), "action": "list_events"}}
            
            # No parameters or necessary functions available
            return {"response": "I understand you want to cancel an event. Please provide the event ID or the date and time of the event you'd like to cancel.", "function_result": {"action": "cancellation_info_needed", "status": "pending", "intent": intent}}
        
        # Handle availability check intents - use the LLM to check if it's asking for earliest time
        elif intent == "availability_intent":
            self._logger.info("Processing availability intent")
            
            # Check if user is asking for earliest time
            is_earliest_request = await self._detect_earliest_time_request(message)
            
            # If checking for availability, extract date/time parameters
            get_available_slots_func = functions.get("check_availability")
            if get_available_slots_func:
                try:
                    # First, try to extract date/time using LLM for the current message
                    dt_params = await self._extract_datetime_with_llm(message)
                    date_param = dt_params.get("date")
                    
                    # If LLM didn't find a date, try ISO format as fallback
                    if not date_param:
                        iso_result = self._extract_iso_datetime(message)
                        if iso_result and "date" in iso_result:
                            date_param = iso_result["date"]
                    
                    # If still no date, try to extract from history using LLM
                    if not date_param:
                        for i, msg in enumerate(reversed(messages[:-1])):
                            if i >= 5:  # Only check last 5 messages
                                break
                            if msg["role"] == "user":
                                hist_dt_params = await self._extract_datetime_with_llm(msg["content"])
                                if hist_dt_params and "date" in hist_dt_params:
                                    date_param = hist_dt_params["date"]
                                    break
                    
                    # If no date found or specified, default to today
                    if not date_param:
                        today = datetime.now()
                        date_param = today.strftime("%Y-%m-%d")
                        self._logger.info(f"No date found, defaulting to today: {date_param}")
                    
                    # Get duration from parameters, history, or default to 30 minutes
                    duration_param = self._extract_duration_param(messages)
                    
                    # Call the availability check function
                    self._logger.info(f"Checking availability for date: {date_param}, duration: {duration_param}, earliest: {is_earliest_request}")
                    availability_result = await get_available_slots_func(
                        date_str=date_param,
                        duration=duration_param,
                        earliest=is_earliest_request
                    )
                    
                    if availability_result.get("status") == "success":
                        available_slots = availability_result.get("available_slots", [])
                        
                        if available_slots:
                            # Format the available slots for display
                            if len(available_slots) == 1:
                                slot = available_slots[0]
                                response_text = f"I found one available slot on {slot.get('formatted_date')}: {slot.get('formatted_time')}. Would you like to book this time?"
                            else:
                                slots_text = "\n".join([
                                    f"- {slot.get('formatted_date')} at {slot.get('formatted_time')}"
                                    for slot in available_slots[:5]  # Limit to first 5 slots
                                ])
                                
                                if len(available_slots) > 5:
                                    slots_text += f"\n\n...and {len(available_slots) - 5} more available times."
                                    
                                response_text = f"I found these available slots:\n\n{slots_text}\n\nWould you like to book one of these times?"
                        else:
                            response_text = f"I'm sorry, but I couldn't find any available slots on {date_param}. Would you like to check a different day?"
                            
                        return {
                            "response": response_text, 
                            "function_result": availability_result, 
                            "action_taken": "get_available_slots", 
                            "intent": intent,
                            "details": availability_result
                        }
                    else:
                        error_msg = availability_result.get("message", "Unknown error")
                        return {
                            "response": f"I couldn't check availability: {error_msg}", 
                            "function_result": availability_result, 
                            "action_taken": "get_available_slots", 
                            "intent": intent,
                            "details": availability_result
                        }
                        
                except Exception as e:
                    self._logger.error(f"Error checking availability: {e}", exc_info=True)
                    error_details = {
                        "status": "error",
                        "message": str(e)
                    }
                    return {
                        "response": f"I encountered an error while checking availability: {str(e)}", 
                        "function_result": error_details, 
                        "action_taken": "error", 
                        "intent": intent,
                        "details": error_details
                    }
            
            # No check_availability function available
            error_details = {
                "status": "error",
                "message": "check_availability function not found"
            }
            return {
                "response": "I understand you want to check availability, but I don't have access to the calendar system right now.", 
                "function_result": error_details, 
                "action_taken": "error", 
                "intent": intent,
                "details": error_details
            }
            
            # For other intents, fall back to regular conversation
        try:
            # Convert to LangChain messages
            lc_messages = self._convert_to_langchain_messages(messages)
            
            # Use the regular LLM for non-booking intents
            try:
                chat_response = await self.llm.agenerate([lc_messages])
                
                # Extract the text from the chat response using our helper
                self._logger.debug(f"Chat response type: {type(chat_response)}")
                
                response_text = ""
                if hasattr(chat_response, 'generations') and len(chat_response.generations) > 0:
                    gen = chat_response.generations[0][0]
                    response_text = self._extract_response_content(gen)
            except Exception as llm_error:
                self._logger.error(f"Error generating response from LLM: {llm_error}")
                self._logger.error(traceback.format_exc())
                response_text = "I'm sorry, I'm having trouble understanding your request. Could you please rephrase it?"
                    
            # If the response is empty for some reason, provide a generic response
            if not response_text:
                response_text = "I understand. How else can I assist you with your calendar?"
            
            # Create results dictionary with all required fields
            result = {
                "response": response_text, 
                "function_result": {"intent": intent, "action": "conversation", "status": "completed"}, 
                "action_taken": "conversation_only", 
                "intent": intent,
                "details": {"intent": intent, "action": "conversation", "status": "completed"}
            }
            
            return result
        except Exception as e:
            self._logger.error(f"Error in regular conversation processing: {e}", exc_info=True)
            error_details = {
                "status": "error", 
                "message": str(e), 
                "action": "conversation_error"
            }
            return {
                "response": "I'm sorry, but I encountered an error while processing your message. Please try again.",
                "function_result": error_details,
                "action_taken": "error", 
                "intent": intent,
                "details": error_details
            }

    def _convert_to_langchain_messages(self, messages: List[Dict[str, str]]) -> List[Any]:
        """
        Convert standard message format to LangChain message objects.
        
        Args:
            messages: A list of messages in the format {"role": "...", "content": "..."}
            
        Returns:
            A list of LangChain message objects
        """
        from langchain.schema import SystemMessage, HumanMessage, AIMessage
        
        lc_messages = []
        try:
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    # Ensure content is a string
                    if not isinstance(content, str):
                        content = str(content)
                    
                    if role == "system":
                        lc_messages.append(SystemMessage(content=content))
                    elif role == "user":
                        lc_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    # Ignore other message types like 'function' for now
                else:
                    self._logger.warning(f"Skipping message with invalid format: {msg}")
        except Exception as e:
            self._logger.error(f"Error converting messages to LangChain format: {e}")
            self._logger.error(f"Traceback: {traceback.format_exc()}")
            # Return what we have so far, even if incomplete
            
        return lc_messages

    def get_default_system_message(self) -> str:
        """
        Get the default system message for Cal.com booking.
        
        Returns:
            The system prompt
        """
        system_prompt = """
        You are a helpful AI assistant that helps people schedule meetings on their calendar through Cal.com. 
        Always be brief, helpful, and friendly in your responses.
        
        IMPORTANT GUIDELINES:
        
        1. ALWAYS VERIFY AVAILABILITY: Never provide time slots without checking actual availability. Do not make up times.
        
        2. IMMEDIATE RESPONSES: When asked about earliest available slots, provide the information immediately without asking for confirmation.
        
        3. FORMAT TIMES PROPERLY: Display times in a user-friendly format, including the time zone. For example, "Thursday, May 15, 2025 at 8:00 AM PST".
        
        4. HANDLE ERRORS GRACEFULLY: If the calendar service is unavailable or returns errors, be transparent about the issue.
        
        5. GATHERING INFORMATION: For scheduling, you need:
           - Duration (15 or 30 minutes)
           - Date (get specific date)
           - Time zone (convert to PST if needed)
        
        6. BOOKING PROCESS:
           - First verify the time is actually available
           - For confirmed bookings, collect name and email
           - Keep responses concise and professional
        
        7. TIME ZONE HANDLING: Default to PST if user mentions PST, Pacific, or California time. Otherwise, ask for clarification.
        
        8. ERROR SITUATIONS: If no slots are available on a requested date, suggest checking nearby dates instead.
        
        Remember to act as a professional scheduling assistant and maintain a helpful, efficient conversation.
        """
        return system_prompt 

    async def initialize(self):
        """
        Initialize the function caller with system prompt and predefined functions.
        """
        try:
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=0.5,
                openai_api_key=self.api_key
            )
            
            self.logger.info(f"Initialized LangChain ChatOpenAI client with model {self.model}")
            
            # Define the default system message
            self.system_message = self.get_default_system_message()
            
            # Initialize LangChain's tool calling interface
            self.system_prompt = HumanMessage(content=self.system_message)
            
            # Create function caller with the functions
            self.function_caller = LLMChain(
                llm=self.llm,
                prompt=ChatPromptTemplate.from_messages([self.system_prompt, MessagesPlaceholder(variable_name="messages")])
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize function caller: {e}")
            return False 

    def _extract_response_content(self, response):
        """
        Extract content from various response structures.
        
        Args:
            response: The response object to extract content from
            
        Returns:
            The extracted content as a string
        """
        try:
            # Try common patterns for content extraction
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            elif hasattr(response, 'text'):
                return response.text
            elif isinstance(response, dict) and 'text' in response:
                return response['text']
            elif hasattr(response, 'message'):
                if hasattr(response.message, 'content'):
                    return response.message.content
                elif isinstance(response.message, dict) and 'content' in response.message:
                    return response.message['content']
            elif isinstance(response, str):
                return response
                
            self._logger.warning(f"Could not extract content from response type: {type(response)}")
            self._logger.debug(f"Response structure: {dir(response) if hasattr(response, '__dict__') else str(response)}")
            return "I'm not sure how to respond to that."
        except Exception as e:
            self._logger.error(f"Error extracting content: {e}")
            self._logger.error(f"Traceback: {traceback.format_exc()}")
            return "I encountered an error while processing your request." 

    def _is_simple_time_response(self, message: str) -> bool:
        """
        Check if the message is a simple time response (e.g., "3pm", "15:00").
        
        Args:
            message: The message to check
            
        Returns:
            True if the message is a simple time response, False otherwise
        """
        message = message.lower().strip()
        
        # Check for simple time formats
        return bool(
            re.match(r'^(\d{1,2})[\\.:]?(\d{2})?(\s*[ap]m)?$', message) or  # 3pm, 3:00, 15:00
            re.match(r'^(at\s+)?(\d{1,2})(\s*[ap]m)$', message) or         # at 3pm
            re.match(r'^(morning|afternoon|evening)$', message) or         # afternoon
            re.match(r'^\s*(yes|sure|okay|ok|fine|good|excellent)\s+(\d{1,2})[\\.:]?(\d{2})?(\s*[ap]m)?\s*$', message)  # yes 3pm
        )
    
    def _normalize_time_format(self, time_str: str) -> str:
        """
        Normalize various time formats to a standard HH:MM format.
        
        Args:
            time_str: A time string in various formats (e.g., "3pm", "15:00", "afternoon", "9:00 AM PST")
            
        Returns:
            Normalized time in HH:MM format
        """
        if not time_str:
            return "00:00"  # Default to midnight if empty
            
        time_str = time_str.lower().strip()
        
        # Handle word-based times
        if time_str == "morning":
            return "09:00"
        elif time_str == "afternoon":
            return "14:00"
        elif time_str == "evening":
            return "18:00"
        
        # Remove timezone indicators
        timezone_indicators = ["pst", "est", "cst", "mst", "gmt", "utc"]
        for tz in timezone_indicators:
            if f" {tz}" in time_str:
                time_str = time_str.replace(f" {tz}", "")
        
        # Remove "at " prefix if present
        if time_str.startswith("at "):
            time_str = time_str[3:].strip()
        
        # Remove "at" anywhere in the string (like "9:00 at AM")
        time_str = time_str.replace(" at ", " ").strip()
        
        # Try to parse various time formats
        am_pm_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', time_str)
        if am_pm_match:
            hour = int(am_pm_match.group(1))
            minute = int(am_pm_match.group(2) or "0")
            am_pm = am_pm_match.group(3).lower()
            
            # Convert to 24-hour format
            if am_pm == "pm" and hour < 12:
                hour += 12
            elif am_pm == "am" and hour == 12:
                hour = 0
                
            return f"{hour:02d}:{minute:02d}"
        
        # Check if it's already in HH:MM format
        time_match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            return f"{hour:02d}:{minute:02d}"
        
        # Handle just hours with am/pm
        hour_match = re.match(r'^(\d{1,2})\s*(am|pm)$', time_str)
        if hour_match:
            hour = int(hour_match.group(1))
            am_pm = hour_match.group(2).lower()
            
            # Convert to 24-hour format
            if am_pm == "pm" and hour < 12:
                hour += 12
            elif am_pm == "am" and hour == 12:
                hour = 0
                
            return f"{hour:02d}:00"
        
        # Handle military time without colon
        military_match = re.match(r'^(\d{3,4})$', time_str)
        if military_match:
            time_digits = military_match.group(1)
            if len(time_digits) == 3:
                time_digits = "0" + time_digits
                
            hour = int(time_digits[:2])
            minute = int(time_digits[2:])
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
        
        # Just a number (assume it's an hour)
        hour_only_match = re.match(r'^(\d{1,2})$', time_str)
        if hour_only_match:
            hour = int(hour_only_match.group(1))
            # If it's just a number between 1-12, assume it's PM unless it's 12
            if 1 <= hour <= 11:
                hour += 12  # Assume PM
            return f"{hour:02d}:00"
        
        # Return as is if we couldn't parse it
        self._logger.warning(f"Could not normalize time format: {time_str}")
        return time_str
    
    def _extract_date_from_history(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract date information from conversation history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Extracted date in YYYY-MM-DD format or None if not found
        """
        # Look through previous messages from newest to oldest
        for message in reversed(messages):
            if message["role"] == "user":
                # First try ISO format
                iso_result = self._extract_iso_datetime(message["content"])
                if iso_result and "date" in iso_result:
                    return iso_result["date"]
                
                # For more complex date extraction, the LLM-based function
                # calling approach should be used in the process_with_function_calling
                # instead of this method
                
        return None
    
    def _extract_booking_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract booking parameters from the messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Dictionary containing extracted booking parameters
        """
        # Start with the most recent message
        last_message = messages[-1]["content"] if messages else ""
        
        # First try the ISO 8601 format (which might still be useful to retain)
        booking_params = self._extract_iso_datetime(last_message)
        
        # If no ISO format found, we'll use the LLM-based approach for natural language
        if not booking_params:
            # This will be handled by the LLM function calling in process_with_function_calling
            # We're moving away from regex-based extraction
            pass
            
        # Add additional parameters if available
        name_match = re.search(r'name(?:\s+is)?(?:\s*[:=]?)?\s*([A-Za-z\s]+)', last_message, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            booking_params["name"] = name
            
        # Extract any duration mentioned
        duration_match = re.search(r'(\d+)\s*(?:min|minute|minutes)', last_message, re.IGNORECASE)
        if duration_match:
            duration = int(duration_match.group(1))
            booking_params["duration"] = duration
            
        return booking_params
    
    def _extract_date_param(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract date parameter from messages specifically for availability checks.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Extracted date in YYYY-MM-DD format or None if not found
        """
        # Start with the most recent message
        last_message = messages[-1]["content"] if messages else ""
        
        # This will be handled by the LLM function calling in process_with_function_calling
        # Moving away from regex-based extraction
        
        # For ISO format we can still use regex since it's a standard format
        iso_result = self._extract_iso_datetime(last_message)
        if iso_result and "date" in iso_result:
            return iso_result["date"]
        
        # For history-based extraction, we could use a loop to check previous messages
        # but that will be handled by the LLM function calling
        
        return None

    def _extract_duration_param(self, messages: List[Dict[str, str]]) -> Optional[int]:
        """
        Extract meeting duration from messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Duration in minutes or None if not found
        """
        # Start with the most recent message
        last_message = messages[-1]["content"] if messages else ""
        
        # Default durations
        default_durations = [15, 30, 60]
        
        # Simple regex for duration in minutes - we keep this for backward compatibility
        duration_match = re.search(r'(\d+)\s*(?:min|minute|minutes)', last_message, re.IGNORECASE)
        if duration_match:
            try:
                duration = int(duration_match.group(1))
                # Make sure it's a reasonable duration
                if duration > 0 and duration <= 180:  # Max 3 hours
                    return duration
            except (ValueError, IndexError):
                pass
        
        # For more complex extractions, we'll use the LLM in process_with_function_calling
        
        # Default to 30 minutes if no explicit duration
        return 30

    async def _detect_earliest_time_request(self, message: str) -> bool:
        """
        Detect if the user is asking for the earliest available time.
        
        Args:
            message: The user message
            
        Returns:
            True if the user is asking for the earliest time, False otherwise
        """
        try:
            # Create a function calling LLM
            function_llm = ChatOpenAI(
                model=self.model,
                temperature=0.1,  # Low temperature for precision
                openai_api_key=self.api_key
            )
            
            # Define the detection function
            function_definitions = [
                {
                    "name": "detect_earliest_time_request",
                    "description": "Detect if the user is asking for the earliest or next available time slot",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_earliest_request": {
                                "type": "boolean",
                                "description": "True if the user is asking for the earliest/next/soonest available time"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence level from 0.0 to 1.0"
                            }
                        },
                        "required": ["is_earliest_request"]
                    }
                }
            ]
            
            # Prepare the model with functions
            function_llm = function_llm.bind(
                functions=function_definitions
            )
            
            # Create system message
            system_message = SystemMessage(content="""
            You are a message intent analyzer. Determine if the user is asking for the earliest available time slot.
            Examples that would be earliest time requests:
            - "What's the earliest time you have?"
            - "When's your next available slot?"
            - "I need the soonest appointment possible"
            - "What's the first opening you have?"
            - "ASAP please"
            
            Examples that would NOT be earliest time requests:
            - "I want to book for 3pm"
            - "Is 2pm on Friday available?"
            - "Let me know what times you have on Tuesday"
            - "Can we meet next week?"
            """)
            
            # Get a response with function calling
            messages = [system_message, HumanMessage(content=message)]
            self._logger.info("Calling LLM to detect earliest time request")
            response = await function_llm.ainvoke(messages)
            
            # Check if the model returned a function call
            if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                function_call = response.additional_kwargs['function_call']
                
                if function_call['name'] == 'detect_earliest_time_request':
                    # Parse the arguments
                    args = json.loads(function_call['arguments'])
                    is_earliest = args.get('is_earliest_request', False)
                    confidence = args.get('confidence', 0.0)
                    
                    self._logger.info(f"Earliest time detection: {is_earliest} (confidence: {confidence})")
                    return is_earliest
            
            # Fall back to False if LLM fails
            return False
            
        except Exception as e:
            self._logger.error(f"Error in earliest time request detection: {str(e)}")
            self._logger.error(traceback.format_exc())
            return False

    def _extract_iso_datetime(self, message: str) -> Dict[str, str]:
        """
        Extract ISO 8601 datetime from a message.
        
        Args:
            message: The message to extract datetime from.
            
        Returns:
            A dictionary with date and time if found, otherwise empty dict
        """
        # Pattern to match ISO 8601 datetime (e.g., 2025-05-14T20:30:00.000Z)
        iso_pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}):00\.000Z'
        match = re.search(iso_pattern, message)
        
        if match:
            date_str = match.group(1)  # YYYY-MM-DD
            time_str = match.group(2)  # HH:MM
            
            self._logger.info(f"Extracted ISO date: {date_str}, time: {time_str} from message")
            return {
                "date": date_str,
                "time": time_str
            }
        return {}

    def _extract_natural_language_datetime(self, message: str) -> Dict[str, str]:
        """
        Extract date and time from natural language expressions like "tomorrow at 2pm".
        
        Args:
            message: The message to extract datetime from.
            
        Returns:
            A dictionary with date and time if found, otherwise empty dict
        """
        # We'll use the function calling capability with LLM to extract date and time
        # This will be implemented using the LLM directly instead of regex patterns
        self._logger.info("Using LLM to extract natural language datetime")
        
        # For now, return empty dict - this will be handled by LLM function calling later
        return {}

    async def _extract_datetime_with_llm(self, message: str) -> Dict[str, str]:
        """
        Use LLM to extract date and time from natural language expressions.
        
        Args:
            message: The message to extract datetime from.
            
        Returns:
            A dictionary with date and time if found, otherwise empty dict
        """
        try:
            # Create a function calling LLM
            function_llm = ChatOpenAI(
                model=self.model,
                temperature=0.1,  # Low temperature for precision
                openai_api_key=self.api_key
            )
            
            # Define the date/time extraction function
            date_today = datetime.today().strftime("%Y-%m-%d")
            function_definitions = [
                {
                    "name": "extract_datetime",
                    "description": "Extract date and time information from text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": f"The date in YYYY-MM-DD format. Today is {date_today}."
                            },
                            "time": {
                                "type": "string",
                                "description": "The time in 24-hour format (HH:MM)"
                            }
                        },
                        "required": []
                    }
                }
            ]
            
            # Prepare the model with functions
            function_llm = function_llm.bind(
                functions=function_definitions
            )
            
            # Create system message for date/time extraction
            system_message = SystemMessage(content="""
            You are a datetime extraction assistant. Extract any date and time information from the user's message.
            If a date is mentioned, convert it to YYYY-MM-DD format.
            If a time is mentioned, convert it to 24-hour format (HH:MM).
            If the date or time is ambiguous or relative (like "tomorrow"), make a reasonable interpretation based on current date context.
            If no date or time is mentioned, do not include those fields in your response.
            """)
            
            # Get a response with function calling
            messages = [system_message, HumanMessage(content=message)]
            self._logger.info("Calling LLM for datetime extraction")
            response = await function_llm.ainvoke(messages)
            
            # Check if the model returned a function call
            if hasattr(response, 'additional_kwargs') and 'function_call' in response.additional_kwargs:
                function_call = response.additional_kwargs['function_call']
                self._logger.info(f"Function call detected: {function_call['name']}")
                
                if function_call['name'] == 'extract_datetime':
                    # Parse the arguments
                    args = json.loads(function_call['arguments'])
                    self._logger.info(f"Extracted datetime: {args}")
                    
                    # Return the extracted date and time
                    result = {}
                    if 'date' in args and args['date']:
                        result['date'] = args['date']
                    if 'time' in args and args['time']:
                        result['time'] = args['time']
                    
                    return result
            
            # If no function call or if it failed, return empty dict
            return {}
                
        except Exception as e:
            self._logger.error(f"Error in LLM datetime extraction: {str(e)}")
            self._logger.error(traceback.format_exc())
            return {} 