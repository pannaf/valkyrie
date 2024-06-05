"""Test the state module."""

from src.state_graph.state import State, update_dialog_stack
from src.assistants.assistant_type import AssistantType


class TestState:
    """Test the state module."""

    def setup_method(self):
        """Setup any state specific initialization."""
        self.initial_stack = [AssistantType.GANDALF]
        self.new_state = AssistantType.ONBOARDING_WIZARD

    def test_update_dialog_stack_add_state(self):
        """Test adding a new state to the stack."""
        expected_stack = [AssistantType.GANDALF, AssistantType.ONBOARDING_WIZARD]
        assert update_dialog_stack(self.initial_stack, self.new_state) == expected_stack

    def test_update_dialog_stack_pop_state(self):
        """Test popping a state from the stack."""
        initial_stack = [AssistantType.GANDALF, AssistantType.ONBOARDING_WIZARD]
        expected_stack = [AssistantType.GANDALF]
        assert update_dialog_stack(initial_stack, "pop") == expected_stack

    def test_update_dialog_stack_none_state(self):
        """Test handling None state."""
        assert update_dialog_stack(self.initial_stack, None) == self.initial_stack

    def test_state_initialization(self):
        """Test the initialization of the State class."""
        state = State(messages=[], user_info={"name": "Test User"}, dialog_state=[AssistantType.GANDALF])
        assert isinstance(state, dict)
        assert "messages" in state
        assert "user_info" in state
        assert "dialog_state" in state
        assert state["user_info"]["name"] == "Test User"
        assert state["dialog_state"] == [AssistantType.GANDALF]
