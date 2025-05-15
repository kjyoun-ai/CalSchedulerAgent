"""
Cal.com API client for the Calendar Agent.
This module handles all interactions with the Cal.com API.
"""

import logging
from typing import Dict, Any, List, Optional
import httpx
from src.utils.config import CAL_API_KEY, CAL_API_URL, logger
import json
import aiohttp
from datetime import datetime, timedelta
import traceback

class CalAPIClient:
    """
    Client for interacting with the Cal.com API.
    """
    
    def __init__(self):
        """Initialize the Cal.com API client."""
        self.logger = logging.getLogger(__name__)
        
        # Use API key from environment variables
        self.api_key = CAL_API_KEY
        self.logger.info(f"Using Cal.com API key: {self.api_key}")
        
        self.api_url = CAL_API_URL
        # Headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        
        # Cache for user data
        self._cached_user = None
        
    async def __aenter__(self):
        """Support for async context manager."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        await self.client.aclose()
    
    async def test_api_connection(self) -> bool:
        """
        Test the connection to the Cal.com API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to get user availability, which is a simple authenticated endpoint
            self.logger.info(self.api_key)
            response = await self.client.get(
                f"{self.api_url}/schedules", 
                params={"apiKey": self.api_key}
            )
            response.raise_for_status()
            self.logger.info("Cal.com API connection test successful")
            return True
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Cal.com API connection test failed with status {e.response.status_code}")
            self.logger.error(f"Response: {e.response.text}")
            return False
        except Exception as e:
            self.logger.error(f"Cal.com API connection test failed: {str(e)}")
            return False
    
    async def get_event_types(self) -> Dict[str, Any]:
        """
        Get all available event types.
        
        Returns:
            A dictionary containing event types
        """
        self.logger.info("Getting event types")
        
        try:
            url = f"{self.api_url}/event-types"
            response = await self.client.get(url, params={"apiKey": self.api_key})
            response.raise_for_status()
            data = response.json()
            
            self.logger.info(f"Retrieved {len(data.get('event_types', []))} event types")
            return data
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Failed to get event types: {e.response.status_code}")
            self.logger.error(f"Response: {e.response.text}")
            return {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {e.response.text}"
            }
        except Exception as e:
            self.logger.error(f"Failed to get event types: {str(e)}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    
    async def get_available_slots(self, event_type_id: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """
        Get available slots for a specific event type.
        
        Args:
            event_type_id: ID of the event type
            start_time: Start time in ISO 8601 format with timezone (e.g., 2025-05-14T00:00:00.000Z)
            end_time: End time in ISO 8601 format with timezone (e.g., 2025-05-21T23:59:59.000Z)
            
        Returns:
            A dictionary containing available slots
        """
        self.logger.info(f"Getting available slots for event type {event_type_id}")
        
        try:
            # Validate that start_time and end_time are in ISO 8601 format
            if not (start_time and end_time and 'T' in start_time and 'T' in end_time):
                raise ValueError(
                    "start_time and end_time must be in ISO 8601 format with timezone (e.g., 2025-05-14T00:00:00.000Z)"
                )
                
            url = f"{self.api_url}/slots"
            params = {
                "eventTypeId": event_type_id,
                "startTime": start_time,
                "endTime": end_time,
                "apiKey": self.api_key
            }
            
            self.logger.debug(f"Requesting slots with params: {params}")
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            slots_count = sum(len(day_slots) for day_slots in data.get("slots", {}).values())
            self.logger.info(f"Retrieved {slots_count} available slots")
            return data
        except ValueError as e:
            self.logger.error(f"Invalid time format: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Failed to get available slots: {e.response.status_code}")
            self.logger.error(f"Response: {e.response.text}")
            return {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {e.response.text}"
            }
        except Exception as e:
            self.logger.error(f"Failed to get available slots: {str(e)}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    
    async def book_event(
        self, 
        event_type_id: str, 
        start_time: str, 
        name: str,
        email: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Book a new event.
        
        Args:
            event_type_id: ID of the event type
            start_time: Start time in ISO 8601 format with timezone (e.g., 2025-05-14T20:30:00.000Z)
            name: Name of the attendee
            email: Email of the attendee
            reason: Optional reason for the meeting
            
        Returns:
            A dictionary containing booking details
        """
        self.logger.info(f"Booking event for {email} at {start_time}")
        
        try:
            # First, get the event type details to determine exact duration
            url = f"{self.api_url}/event-types/{event_type_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"apiKey": self.api_key}) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to get event type details: HTTP {response.status}")
                        # Default to 30 minutes if we can't get details
                        event_duration = 30
                    else:
                        event_data = await response.json()
                        event_duration = event_data.get("length", 30)
                        self.logger.info(f"Retrieved event duration from API: {event_duration} minutes")
            
            # Calculate end time based on start time and exact event duration
            try:
                # Parse the start time and add the duration
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = start_dt + timedelta(minutes=event_duration)
                
                # Format end time as ISO 8601
                end_time = end_dt.strftime('%Y-%m-%dT%H:%M:00.000Z')
                self.logger.info(f"Calculated end time: {end_time} (event duration: {event_duration} minutes)")
            except Exception as e:
                self.logger.error(f"Error calculating end time: {str(e)}")
                # Fallback calculation if parsing fails
                parts = start_time.split('T')
                if len(parts) == 2:
                    date_part = parts[0]
                    time_part = parts[1].split('.')[0]  # Remove milliseconds
                    hour, minute, second = time_part.split(':')
                    new_minute = int(minute) + event_duration
                    new_hour = int(hour)
                    
                    # Handle minute overflow
                    if new_minute >= 60:
                        new_hour += new_minute // 60
                        new_minute = new_minute % 60
                    
                    # Format the new time
                    end_time = f"{date_part}T{new_hour:02d}:{new_minute:02d}:{second}.000Z"
                    self.logger.info(f"Fallback end time calculation: {end_time}")
                else:
                    # Last resort fallback
                    end_time = start_time
                    self.logger.warning(f"Could not calculate end time, using start time as end time")

            booking_url = f"{self.api_url}/bookings"
            
            # Prepare the request payload using the format that works with the API
            payload = {
                "eventTypeId": int(event_type_id),
                "start": start_time,
                "end": end_time,
                "responses": {
                    "name": name,
                    "email": email
                },
                "timeZone": "America/Los_Angeles",  # Using a fixed timezone that works
                "language": "en",
                "metadata": {}
            }
            
            # Add reason if provided
            if reason:
                payload["responses"]["notes"] = reason
                
            # Change from debug to info for better visibility
            self.logger.info(f"Booking payload: {json.dumps(payload)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    booking_url, 
                    params={"apiKey": self.api_key},
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response_status = response.status
                    response_text = await response.text()
                    
                    # Change from debug to info for better visibility
                    self.logger.info(f"Booking response ({response_status}): {response_text}")
                    
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        self.logger.error(f"Invalid JSON response: {response_text}")
                        return {
                            "status": "error",
                            "message": f"Failed to book meeting: Invalid response from server",
                            "details": response_text
                        }
                    
                    if response_status == 200 or response_status == 201:
                        self.logger.info(f"Successfully booked meeting with ID: {response_data.get('uid', 'unknown')}")
                        return response_data
                    else:
                        error_message = response_data.get('message', 'Unknown error')
                        self.logger.error(f"Failed to book meeting: {error_message}")
                        return {
                            "status": "error",
                            "message": f"Failed to book meeting: {error_message}",
                            "details": response_data
                        }
        except Exception as e:
            self.logger.error(f"Exception while booking event: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Exception while booking event: {str(e)}"
            }
    
    async def list_bookings(self, email: Optional[str] = None) -> Dict[str, Any]:
        """
        List all bookings, optionally filtered by attendee email.
        
        Args:
            email: Optional email to filter by
            
        Returns:
            A dictionary containing a list of bookings
        """
        self.logger.info(f"Listing bookings (filter by email: {email})")
        try:
            url = f"{self.api_url}/bookings"
            response = await self.client.get(url, params={"apiKey": self.api_key})
            response.raise_for_status()
            data = response.json()
            bookings = data.get("bookings", [])

            # Filter by attendee email if provided
            if email:
                filtered = []
                for booking in bookings:
                    attendees = booking.get("attendees", [])
                    if any(a.get("email") == email for a in attendees):
                        filtered.append(booking)
                bookings = filtered

            self.logger.info(f"Found {len(bookings)} bookings")
            return {
                "status": "success",
                "bookings": bookings
            }
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Failed to list bookings: {e.response.status_code}")
            self.logger.error(f"Response: {e.response.text}")
            return {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {e.response.text}"
            }
        except Exception as e:
            self.logger.error(f"Failed to list bookings: {str(e)}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    
    async def cancel_booking(self, booking_id: str) -> Dict[str, Any]:
        """
        Cancel a booking.
        
        Args:
            booking_id: The ID of the booking to cancel
            
        Returns:
            A dictionary containing the cancellation status
        """
        self.logger.info(f"Canceling booking {booking_id}")
        
        try:
            url = f"{self.api_url}/bookings/{booking_id}"
            response = await self.client.delete(url, params={"apiKey": self.api_key})
            response.raise_for_status()
            data = response.json() if response.text else {}
            self.logger.info(f"Successfully cancelled booking {booking_id}")
            return {
                "status": "success",
                "booking_id": booking_id,
                "details": data
            }
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Failed to cancel booking: {e.response.status_code}")
            self.logger.error(f"Response: {e.response.text}")
            return {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {e.response.text}",
                "booking_id": booking_id
            }
        except Exception as e:
            self.logger.error(f"Failed to cancel booking: {str(e)}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "booking_id": booking_id
            }
    
    async def reschedule_booking(self, booking_id: str, new_start_time: str) -> Dict[str, Any]:
        """
        Reschedule a booking to a new time.
        
        Args:
            booking_id: ID of the booking to reschedule
            new_start_time: New start time in ISO 8601 format with timezone (e.g., 2025-05-14T20:30:00.000Z)
            
        Returns:
            A dictionary containing the rescheduled booking details
        """
        self.logger.info(f"Rescheduling booking {booking_id} to {new_start_time}")
        
        try:
            url = f"{self.api_url}/bookings/{booking_id}/reschedule"
            payload = {
                "start": new_start_time
            }
            response = await self.client.patch(url, params={"apiKey": self.api_key}, json=payload)
            response.raise_for_status()
            data = response.json() if response.text else {}
            self.logger.info(f"Successfully rescheduled booking {booking_id}")
            return {
                "status": "success",
                "booking_id": booking_id,
                "details": data
            }
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Failed to reschedule booking: {e.response.status_code}")
            self.logger.error(f"Response: {e.response.text}")
            return {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {e.response.text}",
                "booking_id": booking_id
            }
        except Exception as e:
            self.logger.error(f"Failed to reschedule booking: {str(e)}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "booking_id": booking_id
            }
    
    async def get_user(self) -> Dict[str, Any]:
        """
        Get the current user information from Cal.com API.
        Uses cached data if available to avoid unnecessary API calls.
        
        Returns:
            Dictionary with user information or error details
        """
        # Return cached user if available
        if self._cached_user:
            self.logger.info(f"Using cached user: {self._cached_user.get('name')} (ID: {self._cached_user.get('id')})")
            return {
                "status": "success",
                "user": self._cached_user
            }
            
        self.logger.info("Getting user information from API")
        
        try:
            url = f"{self.api_url}/users"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"apiKey": self.api_key}) as response:
                    response_status = response.status
                    response_text = await response.text()
                    
                    if response_status != 200:
                        self.logger.error(f"Failed to get user information: HTTP {response_status}")
                        self.logger.error(f"Response: {response_text}")
                        return {
                            "status": "error",
                            "message": f"Failed to get user information: HTTP {response_status}",
                            "details": response_text
                        }
                    
                    try:
                        data = json.loads(response_text)
                        users = data.get("users", [])
                        
                        if users:
                            # Use the first user in the list
                            user = users[0]
                            self.logger.info(f"Found user: {user.get('name')} (ID: {user.get('id')})")
                            
                            # Cache the user data
                            self._cached_user = user
                            
                            return {
                                "status": "success",
                                "user": user
                            }
                        else:
                            self.logger.error("No users found in response")
                            return {
                                "status": "error",
                                "message": "No users found in response"
                            }
                    except json.JSONDecodeError:
                        self.logger.error(f"Invalid JSON in user response: {response_text}")
                        return {
                            "status": "error",
                            "message": "Invalid JSON in API response",
                            "details": response_text
                        }
        except Exception as e:
            self.logger.error(f"Error getting user information: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting user information: {str(e)}"
            }

    async def get_availability(self, event_type_id: str, start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get availability for an event type between two dates.
        
        Args:
            event_type_id: The ID of the event type
            start_date: The start date (YYYY-MM-DD)
            end_date: The end date (YYYY-MM-DD), defaults to 7 days from start
            
        Returns:
            A dictionary with availability information or error details
        """
        self.logger.info(f"Getting availability for event type {event_type_id} from {start_date}")
        
        # Calculate end_date if not provided
        if not end_date:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = start_datetime + timedelta(days=7)
            end_date = end_datetime.strftime("%Y-%m-%d")
            
        # First, get the user ID
        user_result = await self.get_user()
        if user_result.get("status") != "success":
            self.logger.error(f"Failed to get user information: {user_result.get('message')}")
            return {
                "status": "error",
                "message": "Failed to get user information required for availability check",
                "details": user_result
            }
        
        user = user_result.get("user", {})
        user_id = user.get("id")
        
        if not user_id:
            self.logger.error("User ID not found in response")
            return {
                "status": "error",
                "message": "User ID not found in response",
                "details": user_result
            }
            
        # Now get availability with the user ID
        try:
            url = f"{self.api_url}/availability"
            
            params = {
                "apiKey": self.api_key,
                "dateFrom": start_date,
                "dateTo": end_date,
                "eventTypeId": event_type_id,
                "userId": user_id
            }
            
            self.logger.info(f"Getting availability with params: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response_status = response.status
                    response_text = await response.text()
                    
                    if response_status != 200:
                        self.logger.error(f"Failed to get availability: HTTP {response_status}")
                        self.logger.error(f"Response: {response_text}")
                        return {
                            "status": "error",
                            "message": f"Failed to get availability: HTTP {response_status}",
                            "details": response_text
                        }
                    
                    try:
                        response_data = json.loads(response_text)
                        return {
                            "status": "success",
                            "availability": response_data
                        }
                    except json.JSONDecodeError:
                        self.logger.error(f"Invalid JSON in availability response: {response_text}")
                        return {
                            "status": "error",
                            "message": "Invalid JSON in API response",
                            "details": response_text
                        }
                
        except Exception as e:
            self.logger.error(f"Error getting availability: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting availability: {str(e)}"
            }
            
    async def is_time_available(
        self, 
        event_type_id: str,
        datetime_str: str
    ) -> bool:
        """
        Check if a specific time slot is available.
        
        Args:
            event_type_id: The ID of the event type
            datetime_str: The datetime to check in ISO 8601 format with timezone (e.g., 2025-05-14T20:30:00.000Z)
            
        Returns:
            True if the time is available, False otherwise
        """
        self.logger.info(f"Checking if time {datetime_str} is available for event type {event_type_id}")
        
        try:
            # Parse the requested datetime (in UTC)
            requested_dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            
            # Convert requested time to PST for visual logging (for debugging only)
            pst_offset = timedelta(hours=-8)
            requested_dt_pst = requested_dt + pst_offset
            self.logger.info(f"Checking availability for UTC: {requested_dt.isoformat()} (PST: {requested_dt_pst.isoformat()})")
            
            # Extract the date part for availability lookup
            requested_date = requested_dt.strftime("%Y-%m-%d")
            
            # Get available slots for the date
            availability_result = await self.get_availability(
                event_type_id=event_type_id,
                start_date=requested_date
            )
            
            if availability_result.get("status") != "success":
                self.logger.error(f"Failed to get availability: {availability_result.get('message')}")
                # Be conservative and return False if we couldn't get availability
                return False
            
            # Extract the date ranges from availability response
            availability_data = availability_result.get("availability", {})
            date_ranges = availability_data.get("dateRanges", [])
            
            self.logger.info(f"Checking {len(date_ranges)} date ranges for {datetime_str}")
            
            # Debug information - show all date ranges
            for i, date_range in enumerate(date_ranges):
                start_str = date_range.get("start")
                end_str = date_range.get("end")
                self.logger.info(f"Range {i+1}: {start_str} to {end_str}")
            
            # Properly check if the requested time falls within any of the available ranges
            for date_range in date_ranges:
                start_str = date_range.get("start", "")
                end_str = date_range.get("end", "")
                
                if not start_str or not end_str:
                    continue
                
                try:
                    # Parse the range start and end times
                    range_start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    range_end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    
                    # Check if the requested time falls within this range
                    if range_start <= requested_dt < range_end:
                        self.logger.info(f"Time {datetime_str} falls within range {start_str} to {end_str}")
                        return True
                except ValueError as e:
                    self.logger.error(f"Error parsing date range: {str(e)}")
                    continue
            
            # Fallback check: Consider available if any range exists for the requested date
            # This is a more permissive approach as the API's date ranges might have timezone issues
            has_date_ranges_for_requested_date = False
            for date_range in date_ranges:
                start_str = date_range.get("start", "")
                if start_str and requested_date in start_str:
                    has_date_ranges_for_requested_date = True
                    self.logger.info(f"Found date range for requested date {requested_date}: {start_str}")
                    return True
            
            if not has_date_ranges_for_requested_date:
                self.logger.warning(f"No date ranges found for requested date {requested_date}")
                return False
            
            # Default to False if no ranges match
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking time availability: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Be conservative and return False if we encountered an error
            return False

    async def get_event_type_by_duration(self, duration_minutes: int) -> Dict[str, Any]:
        """
        Get an event type matching the requested duration.
        
        Args:
            duration_minutes: Desired duration in minutes
            
        Returns:
            Dictionary with the matching event type or error details
        """
        self.logger.info(f"Finding event type with duration {duration_minutes} minutes")
        
        try:
            # Get all event types first
            url = f"{self.api_url}/event-types"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"apiKey": self.api_key}) as response:
                    response_status = response.status
                    response_text = await response.text()
                    
                    if response_status != 200:
                        self.logger.error(f"Failed to get event types: HTTP {response_status}")
                        return {
                            "status": "error",
                            "message": f"Failed to get event types: HTTP {response_status}",
                            "details": response_text
                        }
                    
                    try:
                        data = json.loads(response_text)
                        event_types = data.get("event_types", [])
                        
                        # Find an event type matching the requested duration
                        matching_event_types = [et for et in event_types if et.get("length") == duration_minutes]
                        
                        if matching_event_types:
                            # Use the first matching event type
                            event_type = matching_event_types[0]
                            self.logger.info(f"Found matching event type: ID {event_type.get('id')}, title: {event_type.get('title')}")
                            return {
                                "status": "success",
                                "event_type": event_type
                            }
                        
                        # If no exact match, find the closest event type
                        if not matching_event_types and event_types:
                            # Sort by how close the duration is to requested
                            event_types.sort(key=lambda et: abs(et.get("length", 0) - duration_minutes))
                            closest_event_type = event_types[0]
                            self.logger.info(f"No exact match found. Using closest event type: ID {closest_event_type.get('id')}, " 
                                             f"title: {closest_event_type.get('title')}, duration: {closest_event_type.get('length')} minutes")
                            return {
                                "status": "success",
                                "event_type": closest_event_type,
                                "message": f"No exact {duration_minutes}-minute event type found. Using closest match: {closest_event_type.get('length')} minutes"
                            }
                        
                        return {
                            "status": "error",
                            "message": "No event types found"
                        }
                        
                    except json.JSONDecodeError:
                        self.logger.error("Invalid JSON in event types response")
                        return {
                            "status": "error",
                            "message": "Invalid response from calendar service"
                        }
                    
        except Exception as e:
            self.logger.error(f"Error finding event type: {str(e)}")
            return {
                "status": "error",
                "message": f"Error finding event type: {str(e)}"
            }
    
    async def check_available_event_types(self) -> Dict[str, Any]:
        """
        Get a list of available event types.
        
        Returns:
            Dictionary containing available event types or error details
        """
        try:
            self.logger.info("Checking available event types")
            url = f"{self.api_url}/event-types"
            self.logger.info(f"URL: {url}")
            self.logger.info(f"API Key: {self.api_key}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    params={"apiKey": self.api_key}
                ) as response:
                    response_status = response.status
                    response_text = await response.text()
                    self.logger.info(f"Event types response status: {response_status}")
                    self.logger.info(f"Event types response: {response_text}")
                    
                    if response_status != 200:
                        return {
                            "status": "error",
                            "message": f"Failed to get event types: HTTP {response_status}",
                            "details": response_text,
                            "http_code": response_status
                        }
                    
                    try:
                        response_data = json.loads(response_text)
                        event_types = response_data.get("event_types", [])
                        
                        # Extract key details about each event type
                        simplified_event_types = []
                        for et in event_types:
                            simplified_event_types.append({
                                "id": et.get("id"),
                                "slug": et.get("slug"),
                                "title": et.get("title"),
                                "length": et.get("length"),
                                "description": et.get("description"),
                            })
                        
                        return {
                            "status": "success",
                            "event_types": simplified_event_types
                        }
                    except json.JSONDecodeError:
                        return {
                            "status": "error",
                            "message": "Invalid JSON in API response",
                            "details": response_text,
                            "http_code": response_status
                        }
                    
        except Exception as e:
            self.logger.error(f"Error checking event types: {str(e)}")
            return {
                "status": "error",
                "message": f"Error checking event types: {str(e)}",
                "http_code": 500
            } 