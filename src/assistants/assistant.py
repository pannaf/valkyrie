from langchain_core.runnables import Runnable, RunnableConfig


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    # Define the function that calls the model
    def call_model_limit_message_history(self, state):
        messages = []
        for m in state["messages"][::-1]:
            messages.append(m)
            # TODO: change this to be configurable
            if len(messages) >= 20:
                if messages[-1].type != "tool":
                    break
        print(f"LENGTH OF MESSAGES: {len(messages)}")
        response = self.runnable.invoke(
            {
                "messages": messages[::-1],
                "user_info": state.get("user_info"),
                "valid_input": state.get("valid_iput"),
                "dialog_state": state.get("dialog_state"),
            }
        )
        # We return a list, because this will get added to the existing list
        return response

    def __call__(self, state, config: RunnableConfig):
        while True:
            # result = self.runnable.invoke(state)
            result = self.call_model_limit_message_history(state)

            if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
