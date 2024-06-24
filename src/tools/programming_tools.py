import pandas as pd

from langchain_core.tools import tool
from langchain_core.runnables import ensure_config

from src.db.db_utils import set_user_fitness_level_db


@tool
def fetch_exercises(target_muscle_group: str):
    """
    Fetch relevant exercises from the exercise database, based on the target muscle group.

    Parameters:
    - target_muscle_group (str): The muscle group for which exercises should be fetched. can be one of:
      'Lower Back', 'Glutes', 'Legs', 'Calves', 'Hamstrings', 'Hip Flexors', 'Hips', 'Quads', 'Feet', 'Inner Thighs',
      'Adductors', 'Knees', 'Abductors', 'Shoulders', 'Arms', 'Back', 'Chest', 'Forearms', 'Upper body', 'Triceps', 'Neck', 'Biceps',
      'Traps', 'Elbows', 'Wrists', 'Core', 'Full Body', 'Abs', 'Obliques'
    """

    exercises = pd.read_csv("final_exercise_list.csv")
    exercises = exercises[exercises["Target Muscle Groups"] == target_muscle_group].sample(10)

    return exercises.to_dict(orient="records")


@tool
def set_user_fitness_level(fitness_level: str):
    """
    Set the user's fitness level. Can be one of: 'beginner', 'intermediate', 'advanced'.
    You should infer this information based on the user's activity volume which you can get from the user_activities table.

    Parameters:
    - fitness_level (str): The user's fitness level.
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    set_user_fitness_level_db(user_id, fitness_level)

    print(f"Successfully updated 'onboarded' to 'true' for user {user_id}")

    return ""
