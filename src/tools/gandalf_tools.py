from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import ensure_config

from src.db.db_utils import set_user_onboarded_db


class ToOnboardingWizard(BaseModel):
    """Transfers work to a specialized assistant to handle tasks associated with setting user profile info, including what activities
    the user prefers, where they workout, how often they workout, how long they typically workout for, any constraints they have, what
    their fitness level is, how much they weight and what their goal weight is.
    """

    request: str = Field(description="Any necessary follup questions the onboarding wizard should clarify before proceeding.")


class ToGoalWizard(BaseModel):
    """Transfers work to a specialized assistant to handle goal-setting tasks, except the onboarding wizard handles marking the user's goal weight."""

    request: str = Field(description="Any necessary follup questions the goal wizard should clarify before proceeding.")


class ToProgrammingWizard(BaseModel):
    """Transfers work to a specialized assistant to handle programming tasks, including setting up a workout plan and tracking progress."""

    request: str = Field(description="Any necessary follup questions the programming wizard should clarify before proceeding.")


class ToVWizard(BaseModel):
    """Transfers work to a specialized assistant to answer any personal questions about V."""

    request: str = Field(description="Any necessary follup questions the v wizard should clarify before proceeding.")


@tool
def set_user_onboarded():
    """
    Set the user's onboarded status to True.
    Once the user's profile and goals are set, this tool should be called to indicate that the user has completed the onboarding process.
    """

    cfg = ensure_config()
    configuration = cfg.get("configurable", {})
    user_id = configuration.get("user_id")
    if not user_id:
        raise ValueError("User ID is not set in the configuration")

    set_user_onboarded_db(user_id)

    return f"Successfully updated 'onboarded' to 'true' for user {user_id}"
