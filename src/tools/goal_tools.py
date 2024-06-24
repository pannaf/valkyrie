import uuid

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

    return f"Fetched user goals for user_id {user_id=} : {goals=}"


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

    return f"Created a new goal with id {goal_id=} for user {user_id=}"


@tool
def create_goal(goal_type: str, description: str, end_date: str, notes: str):
    """
    Create a new goal for the user in the database.

    Parameters:
     - goal_type (str) : The type of goal that is being set. Infer this from the user's goal description.
     - description (str) : The description of the goal.
     - end_date (str) : The date by which the goal should be completed. In the DB this is stored as a DATE type.
     - notes (str) : Any additional notes about the goal. For example, why the user is setting this goal, if it's a difficult goal for them, if it is a process goal supporting another goal, etc.

    The schema for the goals table is as follows:
     - goal_type TEXT # e.g., 'strength', 'endurance', 'flexibility', 'weight_loss', 'weight_gain', 'process'
     - description TEXT # e.g., 'Lose 10 pounds in 2 months', 'Run a 5k in under 30 minutes', 'Increase flexibility in my hamstrings', etc.
     - end_date TEXT
     - notes TEXT # e.g., 'User has a wedding to attend in 2 months', 'User is training for a marathon', 'User thinks this is a realistic goal for them', etc.
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    goal_id = str(uuid.uuid4())
    goal_id = create_empty_goal_db(user_id, goal_id)

    update_goal_db(user_id, goal_id, "goal_type", goal_type)
    update_goal_db(user_id, goal_id, "description", description)
    update_goal_db(user_id, goal_id, "end_date", end_date)
    update_goal_db(user_id, goal_id, "notes", notes)
    update_goal_db(user_id, goal_id, "goal_status", "active")
    update_goal_db(user_id, goal_id, "start_date", "NOW()")

    return f"Successfully created a new goal with {goal_type=} {description=} {end_date=} {notes=} for user {user_id=}"


@tool
def update_goal(goal_id: str, goal_field: str, goal_value: str):
    """
    Updates an existing goal for the user.

    Parameters:
    - goal_id (str): The id of the goal to update. In order to figure out the goal_id, you need to use the tool fetch_goals to identify which goal is being discussed that needs to be updated.
    - goal_field (str): The field in the goal to update.
        Must be one of: 'goal_type', 'description', 'end_date', 'goal_status', 'notes'
    - goal_value (str): The new value to set for the specified field.

    The schema for the goals table is as follows:
     - goal_type TEXT # e.g., 'strength', 'endurance', 'flexibility', 'weight_loss', 'weight_gain', 'process'
     - description TEXT # e.g., 'Lose 10 pounds in 2 months', 'Run a 5k in under 30 minutes', 'Increase flexibility in my hamstrings', etc.
     - end_date TEXT
     - notes TEXT # e.g., 'User has a wedding to attend in 2 months', 'User is training for a marathon', 'User thinks this is a realistic goal for them', etc.
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    update_goal_db(user_id, goal_id, goal_field, goal_value)

    return f"Successfully updated {goal_field=} to {goal_value=} for {goal_id=} and {user_id=}"
