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
        stack = [AssistantType.GANDALF, AssistantType.ONBOARDING_WIZARD]
        expected_stack = [AssistantType.GANDALF]
        assert update_dialog_stack(stack, "pop") == expected_stack

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

    def test_update_dialog_stack_empty(self):
        """Test updating an empty stack."""
        empty_stack = []
        new_state = AssistantType.ONBOARDING_WIZARD
        expected_stack = [AssistantType.ONBOARDING_WIZARD]
        assert update_dialog_stack(empty_stack, new_state) == expected_stack

    def test_update_dialog_stack_multiple_pops(self):
        """Test popping multiple states from the stack."""
        stack = [AssistantType.GANDALF, AssistantType.GOAL_WIZARD, AssistantType.ONBOARDING_WIZARD]
        expected_stack_after_first_pop = [AssistantType.GANDALF, AssistantType.GOAL_WIZARD]
        expected_stack_after_second_pop = [AssistantType.GANDALF]

        assert update_dialog_stack(stack, "pop") == expected_stack_after_first_pop
        assert update_dialog_stack(expected_stack_after_first_pop, "pop") == expected_stack_after_second_pop

    def test_update_dialog_stack_pop_from_empty(self):
        """Test popping from an empty stack."""
        empty_stack = []
        expected_stack = []
        assert update_dialog_stack(empty_stack, "pop") == expected_stack

    def test_update_dialog_stack_add_duplicate_states(self):
        """Test adding duplicate states to the stack."""
        stack = [AssistantType.GANDALF]
        duplicate_state = AssistantType.GANDALF
        expected_stack = [AssistantType.GANDALF, AssistantType.GANDALF]
        assert update_dialog_stack(stack, duplicate_state) == expected_stack

    def test_update_dialog_stack_pop_until_empty(self):
        """Test popping states until the stack is empty."""
        stack = [AssistantType.GANDALF, AssistantType.GOAL_WIZARD]
        expected_stack_after_first_pop = [AssistantType.GANDALF]
        expected_stack_after_second_pop = []

        assert update_dialog_stack(stack, "pop") == expected_stack_after_first_pop
        assert update_dialog_stack(expected_stack_after_first_pop, "pop") == expected_stack_after_second_pop

    def test_update_dialog_stack_with_unexpected_values(self):
        """Test handling unexpected values."""
        stack = [AssistantType.GANDALF, AssistantType.GOAL_WIZARD]
        # ignore unexpected values
        unexpected_value = "invalid_state"
        assert update_dialog_stack(stack, unexpected_value) == stack
        # ignore non-string values
        non_string_pop = 123
        assert update_dialog_stack(stack, non_string_pop) == stack
