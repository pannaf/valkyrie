import uuid

from langchain_core.tools import tool
from langchain_core.runnables import ensure_config

from src.db.db_utils import fetch_user, fetch_user_activities_db, update_user_activities_db, create_empty_activity_db


@tool
def fetch_user_info():
    """
    Fetch all known immutable information about the user: id, name, email, height, date of birth

    Returns:
        The user's information, as described above.
    """
    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    user_profile = fetch_user(user_id)
    return user_profile


@tool
def fetch_user_activities():
    """
    Fetch all known mutable information about the user: activity preferences, workout location, workout frequency, workout duration, workout constraints,
         fitness level, weight, goal weight

    Returns:
        The user's profile information, as described above.
    """
    print("Fetching user profile info.......")
    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    user_profile = fetch_user_activities_db(user_id)
    print(f"Fetched user profile for user_id {user_id}: {user_profile}")

    return f"Fetched user profile for user_id {user_id}: {user_profile}"


@tool
def update_activity(activity_id: str, user_activity_field: str, user_activity_value: str):
    """
    Updates a user's profile information in the user_activities table based on the provided field and value.
    If the field is already set to the provided value, don't use this tool.

    Parameters:
    - activity_id (str): The id of the activity to update. In order to figure out the activity_id, you need to use the tool fetch_user_activities to identify which activity is being discussed that needs to be updated.
    - user_activity_field (str): The field in the user profile to update.
        Must be one of: 'activity_name', 'activity_location', 'activity_duration', 'activity_frequency'
    - user_activity_value (str): The new value to set for the specified field.

    The schema for the user_activities table is as follows:
     - activity_name TEXT # e.g., 'running', 'swimming', 'yoga', 'lifting weights', etc.
     - activity_location TEXT # e.g., 'gym', 'home', 'outdoors', etc.
     - activity_duration TEXT # e.g., '30 minutes', '1 hour', '2 hours', etc.
     - activity_frequency TEXT # e.g., '7 days a week', '3 times a week', 'every other week', etc.
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    update_user_activities_db(user_id, activity_id, user_activity_field, user_activity_value)

    print(f"Successfully updated {user_activity_field} to {user_activity_value} for user {user_id}")

    return f"Successfully updated {activity_id=} as: {user_activity_field=} to {user_activity_value=} for user {user_id=}"


@tool
def create_activity(activity_name: str, activity_location: str, activity_duration: str, activity_frequency: str):
    """
    Create a new activity entry for the user.

    Parameters:
    - activity_name (str): The name of the activity to create. e.g., 'running', 'swimming', 'yoga', 'lifting weights', etc.
    - activity_location (str): The location of the activity. e.g., 'gym', 'home', 'outdoors', etc.
    - activity_duration (str): The duration of the activity. e.g., '30 minutes', '1 hour', '2 hours', etc.
    - activity_frequency (str): The frequency of the activity. e.g., '7 days a week', '3 times a week', 'every other week', etc.
    """
    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    user_activity_id = str(uuid.uuid4())
    user_activity_id = create_empty_activity_db(user_id, user_activity_id)
    print(f"Created a new activity with id {user_activity_id} for user {user_id}")

    update_user_activities_db(user_id, user_activity_id, "activity_name", activity_name)
    update_user_activities_db(user_id, user_activity_id, "activity_duration", activity_duration)
    update_user_activities_db(user_id, user_activity_id, "activity_location", activity_location)
    update_user_activities_db(user_id, user_activity_id, "activity_frequency", activity_frequency)

    return f"Created a new activity with id {user_activity_id=} for user {user_id=} and set {activity_name=}, {activity_duration=}, {activity_location=}, {activity_frequency=}"
