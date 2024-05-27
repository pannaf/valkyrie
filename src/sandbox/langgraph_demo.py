from typing import Annotated

import io
from pathlib import Path
from PIL import Image

from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from dotenv import load_dotenv

load_dotenv()

memory = SqliteSaver.from_conn_string(":memory:")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# setup the tools & the chatbot
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)


graph_path = Path("graph.png")
image_data = io.BytesIO(graph.get_graph().draw_mermaid_png())
image = Image.open(image_data)
image.save(graph_path)

import ipdb

ipdb.set_trace()


config = {"configurable": {"thread_id": "1"}}

USER_INPUT = "Hello, my name is Panna. How are you?"
events = graph.stream({"messages": [("user", USER_INPUT)]}, config=config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()

USER_INPUT = "Remember my name?"
events = graph.stream({"messages": [("user", USER_INPUT)]}, config=config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()

# The only difference is we change the `thread_id` here to "2" instead of "1"
events = graph.stream(
    {"messages": [("user", USER_INPUT)]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


while False:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)
