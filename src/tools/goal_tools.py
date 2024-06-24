import json
import uuid
from typing import Union

from langchain_core.runnables import ensure_config
from langchain_core.tools import tool

from src.db.db_utils import create_empty_goal_db, fetch_goals_db, update_goal_db


@tool
def fetch_goals():
    """
    Fetch all known goals for the user.
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    goals = fetch_goals_db(user_id)
    return goals


@tool
def handle_create_goal():
    """
    Create a new goal for the user.
    """
    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    goal_id = str(uuid.uuid4())
    create_empty_goal_db(user_id, goal_id)
    print(f"Created a new goal with id {goal_id} for user {user_id}")

    return "Awesome sauce"


@tool
def update_goal(goal_id: str, goal_field: str, goal_value: Union[str, int, float, list, dict]):
    """
    Updates a user's goal information in the goals table based on the provided field and value.
    If the field is already set to the provided value, don't use this tool.

    Parameters:
    - goal_id (str): The id of the goal to update. In order to figure out the goal_id, you need to use the tool fetch_goals to identify which goal is being discussed that needs to be updated.
    - goal_field (str): The field in the goal to update.
        Must be one of: 'goal_type', 'description', 'target_value', 'current_value', 'unit', 'start_date', 'end_date', 'goal_status', 'notes', 'last_updated'
    - goal_value (Union[str, int, float, list, dict]): The new value to set for the specified field.
        The type of this value should match the type of the field in the database schema.

    The schema for the goals table is as follows:
     - goal_type TEXT
     - description TEXT
     - target_value TEXT
     - current_value TEXT
     - unit TEXT
     - start_date DATE
     - end_date DATE
     - goal_status TEXT
     - notes TEXT
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    if isinstance(goal_value, dict):
        goal_value = json.dumps(goal_value)

    update_goal_db(user_id, goal_id, goal_field, goal_value)

    print(f"Successfully updated {goal_field} to {goal_value} for user {user_id}")

    return "Awesome sauce"
