import pandas as pd

from langchain_core.tools import tool


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
