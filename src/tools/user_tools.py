import json
from typing import Union

from langchain_core.tools import tool
from langchain_core.runnables import ensure_config

from src.db.db_utils import fetch_user, fetch_user_profile, update_user_profile


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
def fetch_user_profile_info():
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

    user_profile = fetch_user_profile(user_id)
    print(f"Fetched user profile for user_id {user_id}: {user_profile}")
    return user_profile


@tool
def set_user_profile_info(user_profile_field: str, user_profile_value: Union[str, int, float, list, dict]):
    """
    Updates a user's profile information in the user_profiles table based on the provided field and value.
    If the field is already set to the provided value, don't use this tool.

    Be specific with the information you are putting in the user_profile_value. For example, if the user mentions that they do multiple
    activites, you should specify each activity location, duration, etc. separately.

    Parameters:
    - user_profile_field (str): The field in the user profile to update.
        Must be one of: 'activity_preferences', 'workout_location', 'workout_frequency', 'workout_duration', 'workout_constraints',
         'fitness_level', 'weight', 'goal_weight'
    - user_profile_value (Union[str, int, float, list, dict]): The new value to set for the specified field.
        The type of this value should match the type of the field in the database schema.

    The schema for the user_profiles table is as follows:
     - activity_preferences TEXT
     - workout_location TEXT
     - workout_frequency TEXT
     - workout_duration TEXT
     - workout_constraints TEXT
     - fitness_level TEXT
     - weight REAL
     - goal_weight REAL
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    if isinstance(user_profile_value, dict):
        user_profile_value = json.dumps(user_profile_value)

    update_user_profile(user_id, user_profile_field, user_profile_value)

    print(f"Successfully updated {user_profile_field} to {user_profile_value} for user {user_id}")

    return "Awesome sauce"
